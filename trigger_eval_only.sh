log(){ echo "[$(date +'%H:%M:%S')] $*"; }
OASIS_TEST="data/OASIS_test"
MAX_PAIRS_OASIS_TEST=100
PAIRS_K=5
PAIRS_SEED=123


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

make_pairs "$OASIS_TEST"  "$OASIS_TEST/pairs_test.csv" "$MAX_PAIRS_OASIS_TEST" --labels

INFER_ON_PAIRS(){ # args: config ckpt pairs outdir
  local cfg="$1" ckpt="$2" pairs="$3" out="$4"
  python -u scripts/eval_pairs.py \
    --pairs "$pairs" \
    --model "$ckpt" \
    --out "$out"
}

FIND_BEST_CKPT(){ # arg: workdir
  local w="$1"
  shopt -s nullglob globstar

  # 1) Prefer *_best.pth (highest epoch)
  local bests=( "$w"/**/checkpoints/*_best.pth "$w"/*_best.pth )
  if (( ${#bests[@]} )); then
    printf '%s\n' "${bests[@]}" | sort -V | tail -n1
    return
  fi
  
  # 3) Then latest epoch_*.pth by numeric order
  local epochs=( "$w"/**/checkpoints/epoch_*.pth "$w"/epoch_*.pth )
  if (( ${#epochs[@]} )); then
    printf '%s\n' "${epochs[@]}" | sort -V | tail -n1
    return
  fi

  # 2) Then final_model.pth (if present)
  local finals=( "$w"/**/checkpoints/final_model.pth "$w"/final_model.pth )
  if (( ${#finals[@]} )); then
    echo "${finals[0]}"
    return
  fi

}

cfg="configs/default.yaml"

work="checkpoints/"

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