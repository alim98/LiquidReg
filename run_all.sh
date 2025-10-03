#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob


# =======================
# Zero-to-hero orchestrator
# =======================

# ---- constants ----

SEEDS=( 1 )                   # seeds for multiple runs
PAIRS_K=5                     # partners per subject in each CSV
PAIRS_SEED=123                # deterministic sampling for pairs
CONFIG_LIST_FILE="configs/cfgs_to_run.txt"  # optional list of configs to run
REMAKE_PAIRS=0

MAX_PAIRS_OASIS_TEST=100
MAX_PAIRS_OASIS_TRAIN=1000
MAX_PAIRS_OASIS_VAL=100
MAX_PAIRS_IXI=100
MAX_PAIRS_L2R=100

RESUME_FROM_CHECKPOINT=1  # 0/1: resume from best available checkpoint in work dir

# data layout (produced by setup_data.sh)
OASIS_TRAIN="data/OASIS_train"
OASIS_VAL="data/OASIS_val"
OASIS_TEST="data/OASIS_test"
L2R_DIR="data/gen_L2R"
IXI_DIR="data/gen_IXI_eval"   # created by setup_data.sh

# Setup
ENV_NAME="LiquidReg_env2"
PYTHON_VERSION="3.12.11"
SKIP_SETUP=1  # set SKIP_SETUP=1 to skip conda/pip setup

# ---- helpers ----
log(){ echo "[$(date +'%H:%M:%S')] $*"; }

need(){ command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }
need python
need bash
need grep
if [ "${SKIP_SETUP:-1}" -eq 0 ]; then
need conda
fi
need find
need awk
need sed

# ---- 0) Data setup (idempotent) ----

if [ "$SKIP_SETUP" -eq 0 ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "[SETUP] Checking Conda environment..."

    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "[SETUP] Creating conda env: $ENV_NAME with Python $PYTHON_VERSION"
        conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    else
        echo "[SETUP] Conda env $ENV_NAME already exists."
    fi

    echo "[SETUP] Activating environment..."
    conda activate $ENV_NAME

    echo "[SETUP] Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[SKIP] Skipping conda & pip setup (SKIP_SETUP=1); using current environment/modules"
fi

log "Running setup_data.sh ..."
bash setup_data.sh

# ---- 1) Make deterministic pairs (idempotent) ----
make_pairs() {
  local root="$1" out="$2" MAX="$3"
  shift 3                           # everything after MAX are extra args (e.g., --labels)

  if [[ -f "$out" && "${REMAKE_PAIRS:-0}" -eq 0 ]]; then
    log "Pairs already exist: $out"
    return
  fi

  log "Making pairs: $out"

  # Build args array safely
  local opts=()
  [[ -n "$MAX" ]] && opts+=(--max_pairs "$MAX")

  # Pass through extra args from the call site
  python -u scripts/make_pairs.py \
    --root "$root" \
    --out "$out" \
    --k "$PAIRS_K" \
    --seed "$PAIRS_SEED" \
    "${opts[@]}" \
    "$@"
}


# ensure helper exists
if [[ ! -f scripts/make_pairs.py ]]; then
  echo "ERROR: scripts/make_pairs.py not found." >&2
  exit 1
fi

# OASIS splits
make_pairs "$OASIS_TRAIN" "$OASIS_TRAIN/pairs_train.csv" "$MAX_PAIRS_OASIS_TRAIN" --labels
make_pairs "$OASIS_VAL"   "$OASIS_VAL/pairs_val.csv" "$MAX_PAIRS_OASIS_VAL" --labels
make_pairs "$OASIS_TEST"  "$OASIS_TEST/pairs_test.csv" "$MAX_PAIRS_OASIS_TEST" --labels

# IXI subset (if present)
if [[ -d "$IXI_DIR" ]]; then
  make_pairs "$IXI_DIR" "$IXI_DIR/pairs_ixi.csv" "$MAX_PAIRS_IXI"
else
  echo "ERROR: IXI eval folder not found ($IXI_DIR)"
  exit 1
fi

# L2R pairs
if [[ -d "$L2R_DIR" ]]; then
  make_pairs "$L2R_DIR" "$L2R_DIR/pairs_l2r.csv" "$MAX_PAIRS_L2R"
# if [[ ! -f "$L2R_DIR/pairs_val.csv" ]]; then
else
  echo "ERROR: $L2R_DIR/pairs_val.csv not found."
  exit 1
fi

# ---- 2) Discover model configs ----
declare -a CONFIGS
if [[ -f "$CONFIG_LIST_FILE" ]]; then
  mapfile -t CONFIGS < <(grep -v '^[[:space:]]*#' "$CONFIG_LIST_FILE" | sed '/^[[:space:]]*$/d')
else
  mapfile -t CONFIGS < <(find configs -maxdepth 1 \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null | sort || true)
  # include baselines if present
  if [[ -d configs/baselines ]]; then
    mapfile -t BASES < <(find configs/baselines -maxdepth 1 \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null | sort || true)
    CONFIGS+=("${BASES[@]}")
  fi
fi

# fallback to a default
if (( ${#CONFIGS[@]} == 0 )); then
  if [[ -f configs/default.yaml ]]; then
    CONFIGS=(configs/default.yaml)
  else
    echo "ERROR: no config YAMLs found under configs/. Add at least one." >&2
    exit 1
  fi
fi
log "Configs to run: ${#CONFIGS[@]}"

# ---- 3) Detect if train.py supports pairs-CSV ----
set +e
python scripts/train.py -h 2>&1 | grep -q -- "--train_pairs"
SUPPORTS_PAIRS=$?
set -e
if [[ "$SUPPORTS_PAIRS" -eq 0 ]]; then
  USE_PAIRS=1
  log "train.py supports --train_pairs/--val_pairs (pairs-based loader will be used)"
else
  USE_PAIRS=0
  log "train.py does NOT expose --train_pairs/--val_pairs (falling back to legacy loader)"
fi

# ---- Choose launcher (python vs torchrun) ----
NUM_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

# Force single-GPU mode to avoid broken PyTorch module distributed components
log "[DDP] ${NUM_GPUS} GPU(s) detected -> forcing single-GPU mode (module distributed broken)"
LAUNCHER="python -u"
DDP_FLAG=""

# ---- 4) Define how to train/infer ----
TRAIN_CMD(){ # args: config workdir seed
  local cfg="$1" work="$2" seed="$3"
  local resume_args=()
  if (( RESUME_FROM_CHECKPOINT )); then
    local ckpt
    ckpt=$(FIND_BEST_CKPT "$work" || true)
    if [[ -n "${ckpt:-}" && -f "$ckpt" ]]; then
      resume_args=(--resume "$ckpt")
    fi
  fi

  if (( USE_PAIRS )); then
    ${LAUNCHER} scripts/train.py \
      --config "$cfg" \
      --train_pairs "$OASIS_TRAIN/pairs_train.csv" \
      --val_pairs   "$OASIS_VAL/pairs_val.csv" \
      --work_dir "$work" \
      --seed "$seed" \
      ${DDP_FLAG} \
      "${resume_args[@]}"
  else
    ${LAUNCHER} scripts/train.py \
      --config "$cfg" \
      --work_dir "$work" \
      --seed "$seed" \
      ${DDP_FLAG} \
      "${resume_args[@]}"
  fi
}


INFER_ON_PAIRS(){ # args: config ckpt pairs outdir
  local cfg="$1" ckpt="$2" pairs="$3" out="$4"
  python -u scripts/eval_pairs.py \
    --pairs "$pairs" \
    --model "$ckpt" \
    --out "$out"
}

# ---- 5) Utility: find a good checkpoint for a run ----
FIND_BEST_CKPT(){ # arg: workdir
  local w="$1"
  shopt -s nullglob globstar

  # 1) Prefer *_best.pth (highest epoch)
  local bests=( "$w"/**/checkpoints/*_best.pth "$w"/*_best.pth )
  if (( ${#bests[@]} )); then
    printf '%s\n' "${bests[@]}" | sort -V | tail -n1
    return
  fi

  # 2) Then final_model.pth (if present)
  local finals=( "$w"/**/checkpoints/final_model.pth "$w"/final_model.pth )
  if (( ${#finals[@]} )); then
    echo "${finals[0]}"
    return
  fi

  # 3) Then latest epoch_*.pth by numeric order
  local epochs=( "$w"/**/checkpoints/epoch_*.pth "$w"/epoch_*.pth )
  if (( ${#epochs[@]} )); then
    printf '%s\n' "${epochs[@]}" | sort -V | tail -n1
    return
  fi

}


# ensure batch evaluator exists
if [[ ! -f scripts/eval_pairs.py ]]; then
  echo "ERROR: scripts/eval_pairs.py not found." >&2
  exit 1
fi

# ---- 6) Train + Eval loop ----
mkdir -p runs
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg"); name="${name%.*}"

  for seed in "${SEEDS[@]}"; do
    work="runs/${name}/seed${seed}"

    # Train (skip if already done)
    if [[ -f "$work/.done" ]]; then
      log "Skip training (already done): $work"
    else
      log "Training $name (seed $seed)"
      mkdir -p "$work"
      TRAIN_CMD "$cfg" "$work" "$seed"
      date > "$work/.done"
    fi

    ckpt=$(FIND_BEST_CKPT "$work")
    if [[ -z "$ckpt" ]]; then
      echo "WARNING: No checkpoint found in $work — skipping eval"; continue
    fi
    log "Using checkpoint: $ckpt"

    # Eval: OASIS_test
    out="$work/eval_oasis_test"
    if [[ ! -f "$out/.done" ]]; then
      mkdir -p "$out"
      INFER_ON_PAIRS "$cfg" "$ckpt" "$OASIS_TEST/pairs_test.csv" "$out"
      date > "$out/.done"
    else
      log "Skip OASIS_test eval (already done)"
    fi

    # Eval: L2R (if pairs exist)
    # if [[ -f "$L2R_DIR/pairs_val.csv" ]]; then
    #   out="$work/eval_l2r"
    #   if [[ ! -f "$out/.done" ]]; then
    #     mkdir -p "$out"
    #     INFER_ON_PAIRS "$cfg" "$ckpt" "$L2R_DIR/pairs_val.csv" "$out"
    #     date > "$out/.done"
    #   else
    #     log "Skip L2R eval (already done)"
    #   fi
    # fi

    if [[ -f "$L2R_DIR/pairs_l2r.csv" ]]; then
      out="$work/eval_l2r"
      if [[ ! -f "$out/.done" ]]; then
        mkdir -p "$out"
        INFER_ON_PAIRS "$cfg" "$ckpt" "$L2R_DIR/pairs_l2r.csv" "$out"
        date > "$out/.done"
      else
        log "Skip L2R eval (already done)"
      fi
    fi

    # Eval: IXI (if subset exists)
    if [[ -f "$IXI_DIR/pairs_ixi.csv" ]]; then
      out="$work/eval_ixi"
      if [[ ! -f "$out/.done" ]]; then
        mkdir -p "$out"
        INFER_ON_PAIRS "$cfg" "$ckpt" "$IXI_DIR/pairs_ixi.csv" "$out"
        date > "$out/.done"
      else
        log "Skip IXI eval (already done)"
      fi
    fi

  done
done

# ---- 7) Tiny overall summary (prints last-line means if present) ----
summarize_dir(){
  local d="$1" tag="$2"
  if [[ -f "$d/summary.txt" ]]; then
    local ncc=$(grep -E '^NCC:' "$d/summary.txt"   | awk '{print $2, $3, $4}')      # mean ± std
    local fld=$(grep -E '^Foldings' "$d/summary.txt" | awk '{print $3, $4, $5}')
    local dsc=$(grep -E '^Dice' "$d/summary.txt"    | awk '{print $3, $4, $5}')
    printf "%-40s | NCC %-14s | Fold %-14s | Dice %-14s\n" "$tag" "$ncc" "$fld" "$dsc"
  fi
}

echo ""
echo "================ Overall summary ================"
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg"); name="${name%.*}"
  for seed in "${SEEDS[@]}"; do
    base="runs/${name}/seed${seed}"
    summarize_dir "$base/eval_oasis_test" "[$name s$seed] OASIS_test"
    [[ -d "$base/eval_l2r" ]] && summarize_dir "$base/eval_l2r" "[$name s$seed] L2R"
    [[ -d "$base/eval_ixi" ]] && summarize_dir "$base/eval_ixi" "[$name s$seed] IXI"
  done
done

log "All done. Explore runs/<model>/seed*/ for logs, checkpoints, and eval outputs."
