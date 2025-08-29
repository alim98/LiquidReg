#!/usr/bin/env bash
set -euo pipefail
# Verbose shell tracing only if VERBOSE=1
if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi
shopt -s globstar nullglob

# ========== Config (override via env) ==========
NEURITE_URL="${NEURITE_URL:-https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar}"

# L2R (generalization set)
L2R_ZIP_URL="${L2R_ZIP_URL:-https://cloud.imi.uni-luebeck.de/s/MSkzxTJTrtfZY5e/download/L2R_2021_Task3_validation.zip}"
L2R_PAIRS_URL="${L2R_PAIRS_URL:-https://cloud.imi.uni-luebeck.de/s/3QCzQ3PsrFxj8eA/download/pairs_val.csv}"

# IXI (generalization set)
IXI_T1_URL="${IXI_T1_URL:-https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar}"
IXI_META_URL="${IXI_META_URL:-https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls}"

# Pretrained (optional)
MODEL_ID="${MODEL_ID:-1T9qYPC0vsAxHHKMzvOOZzvLhLSLiyz0_}"

# Filenames
TAR="${TAR:-neurite-oasis.v1.0.tar}"
L2R_ZIP="${L2R_ZIP:-L2R_2021_Task3_validation.zip}"
L2R_PAIRS="${L2R_PAIRS:-pairs_val.csv}"
IXI_T1_TAR="${IXI_T1_TAR:-IXI-T1.tar}"
IXI_META_FILE="${IXI_META_FILE:-IXI.xls}"

# Directories
DATA_DIR="${DATA_DIR:-data}"
OASIS_STAGE_DIR="${OASIS_STAGE_DIR:-$DATA_DIR/OASIS_stage}"   # temp after extract
OASIS_STORE_DIR="${OASIS_STORE_DIR:-$DATA_DIR/OASIS_store}"   # canonical home (for symlink mode)
OASIS_TRAIN_DIR="${OASIS_TRAIN_DIR:-$DATA_DIR/OASIS_train}"
OASIS_VAL_DIR="${OASIS_VAL_DIR:-$DATA_DIR/OASIS_val}"
OASIS_TEST_DIR="${OASIS_TEST_DIR:-$DATA_DIR/OASIS_test}"
GEN_L2R_DIR="${GEN_L2R_DIR:-$DATA_DIR/gen_L2R}"
GEN_IXI_DIR="${GEN_IXI_DIR:-$DATA_DIR/gen_IXI}"

# Split controls
VAL_FRAC="${VAL_FRAC:-0.15}"
TEST_FRAC="${TEST_FRAC:-0.15}"
VAL_COUNT="${VAL_COUNT:-0}"     # overrides VAL_FRAC if >0
TEST_COUNT="${TEST_COUNT:-0}"   # overrides TEST_FRAC if >0
VAL_SEED="${VAL_SEED:-11}"
TEST_SEED="${TEST_SEED:-42}"
MOVE_MODE="${MOVE_MODE:-1}"     # 1=move real dirs; 0=symlink to OASIS_STORE_DIR

# Behavior flags (0/1)
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
REDO_LEAKAGE="${REDO_LEAKAGE:-0}"   # redo removal of IDs 438–457
REDO_SPLIT="${REDO_SPLIT:-0}"       # rebuild OASIS train/val/test
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
SKIP_PIP="${SKIP_PIP:-1}"           # default skip pip (set 0 to install)
USE_GDOWN="${USE_GDOWN:-0}"
DOWNLOAD_IXI="${DOWNLOAD_IXI:-1}"

# IXI
IXI_EVAL_DIR="${IXI_EVAL_DIR:-$DATA_DIR/gen_IXI_eval}"
IXI_MAX_SUBJECTS="${IXI_MAX_SUBJECTS:-120}"   # change here to keep more/less
IXI_SAMPLE_SEED="${IXI_SAMPLE_SEED:-2025}"

# ========== Helpers ==========
need_cmd(){ command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
say(){ echo "[$(date +'%H:%M:%S')] $*"; }
download_if_needed(){ # url out
  local url="$1" out="$2"
  if [[ -f "$out" && "$FORCE_DOWNLOAD" != "1" ]]; then
    say "Found $out — skip download"
  else
    say "Downloading $out"
    wget -nv -c -O "$out" "$url"
  fi
}
list_dirs(){ find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort; }
count_dirs(){ find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '; }

need_cmd tar; need_cmd unzip; need_cmd wget; need_cmd sha256sum
if [[ "$USE_GDOWN" == "1" ]]; then command -v gdown >/dev/null 2>&1 || true; fi
if [[ "$MOVE_MODE" == "0" ]]; then need_cmd realpath; fi

# 0) Deps
if [[ "$SKIP_PIP" != "1" ]]; then say "pip install -r requirements.txt"; pip install -r requirements.txt; else say "SKIP_PIP=1"; fi

# 1) Make dirs
mkdir -p "$DATA_DIR" "$OASIS_STAGE_DIR" "$OASIS_STORE_DIR" "$OASIS_TRAIN_DIR" "$OASIS_VAL_DIR" "$OASIS_TEST_DIR" "$GEN_L2R_DIR" "$GEN_IXI_DIR"

# 2) Downloads
download_if_needed "$NEURITE_URL" "$TAR"
download_if_needed "$L2R_ZIP_URL" "$L2R_ZIP"
download_if_needed "$L2R_PAIRS_URL" "$L2R_PAIRS"
if [[ "$DOWNLOAD_IXI" == "1" ]]; then
  download_if_needed "$IXI_T1_URL" "$GEN_IXI_DIR/$IXI_T1_TAR"
  download_if_needed "$IXI_META_URL" "$GEN_IXI_DIR/$IXI_META_FILE" || true
fi

# 3) Extract OASIS -> STAGE (skip if STAGE already populated unless FORCE_EXTRACT)
stage_has=$(count_dirs "$OASIS_STAGE_DIR")
if (( stage_has == 0 || FORCE_EXTRACT == 1 )); then
  say "Extracting OASIS into $OASIS_STAGE_DIR"
  tar xf "$TAR" -C "$OASIS_STAGE_DIR"
else
  say "OASIS already staged — skip extract"
fi

# 4) Remove IDs overlapping with L2R (438–457) from STAGE and STORE
remove_overlap_ids(){
  local base="$1"
  [[ -d "$base" ]] || return 0
  pushd "$base" >/dev/null
  for sid in {438..457}; do
    d="OASIS_OAS1_$(printf "%04d" "$sid")_MR1"
    [[ -d "$d" ]] && { say "  - removing $base/$d"; rm -rf -- "$d"; }
  done
  popd >/dev/null
}
if [[ "$REDO_LEAKAGE" == "1" || ! -f "$OASIS_STAGE_DIR/.l2r_removed" ]]; then
  say "Removing potential L2R overlap subjects (438–457)…"
  remove_overlap_ids "$OASIS_STAGE_DIR"
  remove_overlap_ids "$OASIS_STORE_DIR"
  date > "$OASIS_STAGE_DIR/.l2r_removed"
else
  say "Leakage removal previously done — skip (set REDO_LEAKAGE=1 to redo)"
fi

# 5) Build OASIS train/val/test
split_exists=$(( $(count_dirs "$OASIS_TRAIN_DIR") + $(count_dirs "$OASIS_VAL_DIR") + $(count_dirs "$OASIS_TEST_DIR") ))
if (( split_exists > 0 && REDO_SPLIT != 1 )); then
  say "OASIS split already exists — skip (set REDO_SPLIT=1 to rebuild)"
else
  say "Creating OASIS train/val/test (MOVE_MODE=$MOVE_MODE, VAL_FRAC=$VAL_FRAC, TEST_FRAC=$TEST_FRAC)…"

  # If rebuilding, collect everything back
  if (( REDO_SPLIT == 1 )); then
    if (( MOVE_MODE == 1 )); then
      say "Gathering existing subjects back to STAGE for re-split…"
      for d in "$OASIS_TRAIN_DIR" "$OASIS_VAL_DIR" "$OASIS_TEST_DIR"; do
        mapfile -t ds < <(list_dirs "$d")
        for s in "${ds[@]}"; do mv "$d/$s" "$OASIS_STAGE_DIR/" 2>/dev/null || true; done
      done
    else
      say "Clearing old symlinks from train/val/test…"
      for d in "$OASIS_TRAIN_DIR" "$OASIS_VAL_DIR" "$OASIS_TEST_DIR"; do
        find "$d" -mindepth 1 -maxdepth 1 -type l -exec rm -f {} + 2>/dev/null || true
      done
    fi
  fi

  # Ensure canonical storage for symlink mode
  if (( MOVE_MODE == 0 )); then
    # Move any staged subjects into STORE (canonical), keep STORE contents
    mapfile -t stage_subs < <(list_dirs "$OASIS_STAGE_DIR")
    for s in "${stage_subs[@]}"; do
      [[ -d "$OASIS_STORE_DIR/$s" ]] || mv "$OASIS_STAGE_DIR/$s" "$OASIS_STORE_DIR/"
    done
  fi

  # Determine pool to split
  declare -a pool
  if (( MOVE_MODE == 1 )); then
    mapfile -t pool < <(list_dirs "$OASIS_STAGE_DIR")
  else
    mapfile -t pool < <(list_dirs "$OASIS_STORE_DIR")
  fi
  total=${#pool[@]}
  if (( total < 3 )); then
    say "Not enough subjects ($total) to split — aborting."; exit 1
  fi

  # Sizes
  if (( TEST_COUNT > 0 )); then n_test="$TEST_COUNT"; else n_test=$(awk -v n="$total" -v f="$TEST_FRAC" 'BEGIN{printf("%d", (n*f)+0.5)}'); fi
  rem=$(( total - n_test )); if (( n_test < 1 )); then n_test=1; rem=$(( total-1 )); fi
  if (( VAL_COUNT > 0 )); then n_val="$VAL_COUNT"; else n_val=$(awk -v n="$rem" -v f="$VAL_FRAC" 'BEGIN{printf("%d", (n*f)+0.5)}'); fi
  if (( n_val < 1 )); then n_val=1; fi
  if (( n_test + n_val >= total )); then n_val=$(( total - n_test - 1 )); fi

  choose_top_n(){ # seed N, reads ${pool[@]}
    local seed="$1" n="$2"
    for s in "${pool[@]}"; do
      h=$(printf "%s" "${seed}:${s}" | sha256sum | awk '{print $1}')
      printf "%s %s\n" "$h" "$s"
    done | sort | head -n "$n" | awk '{print $2}'
  }
  mapfile -t pick_test < <(choose_top_n "$TEST_SEED" "$n_test")

  # Build remaining pool for val
  declare -A is_test=()
  for s in "${pick_test[@]}"; do is_test["$s"]=1; done
  rem_list=()
  for s in "${pool[@]}"; do [[ ${is_test["$s"]+x} ]] || rem_list+=("$s"); done
  pool=("${rem_list[@]}")  # reuse pool var for val
  mapfile -t pick_val < <( # same chooser on reduced pool
    for s in "${pool[@]}"; do
      h=$(printf "%s" "${VAL_SEED}:${s}" | sha256sum | awk '{print $1}')
      printf "%s %s\n" "$h" "$s"
    done | sort | head -n "$n_val" | awk '{print $2}'
  )

  # Routing helpers
  move_or_link(){
    local src="$1" dst="$2"
    if (( MOVE_MODE == 1 )); then
      mv "$src" "$dst"
    else
      ln -s "$(realpath --relative-to="$(dirname "$dst")" "$src")" "$dst"
    fi
  }

  # Route TEST
  for s in "${pick_test[@]}"; do
    if (( MOVE_MODE == 1 )); then
      move_or_link "$OASIS_STAGE_DIR/$s" "$OASIS_TEST_DIR/$s"
    else
      move_or_link "$OASIS_STORE_DIR/$s" "$OASIS_TEST_DIR/$s"
    fi
  done

  # Route VAL
  for s in "${pick_val[@]}"; do
    if (( MOVE_MODE == 1 )); then
      move_or_link "$OASIS_STAGE_DIR/$s" "$OASIS_VAL_DIR/$s"
    else
      move_or_link "$OASIS_STORE_DIR/$s" "$OASIS_VAL_DIR/$s"
    fi
  done

  # Route remaining -> TRAIN
  if (( MOVE_MODE == 1 )); then
    mapfile -t leftover < <(list_dirs "$OASIS_STAGE_DIR")
    for s in "${leftover[@]}"; do
      move_or_link "$OASIS_STAGE_DIR/$s" "$OASIS_TRAIN_DIR/$s"
    done
  else
    # leftover = STORE minus (test ∪ val)
    declare -A picked=()
    for s in "${pick_test[@]}"; do picked["$s"]=1; done
    for s in "${pick_val[@]}"; do picked["$s"]=1; done
    mapfile -t store_subs < <(list_dirs "$OASIS_STORE_DIR")
    for s in "${store_subs[@]}"; do
      [[ ${picked["$s"]+x} ]] || ln -s "$(realpath --relative-to="$OASIS_TRAIN_DIR" "$OASIS_STORE_DIR/$s")" "$OASIS_TRAIN_DIR/$s"
    done
  fi
fi

# 6) L2R generalization staging
if [[ ! -f "$GEN_L2R_DIR/.ready" ]]; then
  say "Staging L2R into $GEN_L2R_DIR"
  unzip -oq "$L2R_ZIP" -d "$GEN_L2R_DIR"
  cp -n "$L2R_PAIRS" "$GEN_L2R_DIR/" || true
  date > "$GEN_L2R_DIR/.ready"
else
  say "L2R already staged."
fi

# 7) IXI generalization staging (download & extract if requested)
if [[ "$DOWNLOAD_IXI" == "1" ]]; then
  if [[ -f "$GEN_IXI_DIR/.extracted" ]]; then
    say "IXI appears extracted — skip"
  else
    # Look for any T1 NIfTIs without using a pipeline (robust under 'set -euo pipefail')
    mapfile -t _ixi_probe < <(find "$GEN_IXI_DIR" -maxdepth 2 -type f \
      \( -name "*T1*.nii" -o -name "*T1*.nii.gz" \) -print -quit 2>/dev/null)

    if (( ${#_ixi_probe[@]} > 0 )); then
      say "IXI NIfTIs present — marking extracted"
      date > "$GEN_IXI_DIR/.extracted"
    elif [[ -f "$GEN_IXI_DIR/$IXI_T1_TAR" ]]; then
      say "Extracting IXI T1 into $GEN_IXI_DIR"
      tar xf "$GEN_IXI_DIR/$IXI_T1_TAR" -C "$GEN_IXI_DIR"
      date > "$GEN_IXI_DIR/.extracted"
    else
      say "IXI T1 archive not found; set DOWNLOAD_IXI=1 to fetch automatically."
    fi
  fi
else
  say "DOWNLOAD_IXI=0 — skipping IXI download."
fi

# --- IXI subset for evaluation (flat copy into one folder) ---
IXI_EVAL_DIR="${IXI_EVAL_DIR:-$DATA_DIR/gen_IXI_eval}"
IXI_MAX_SUBJECTS="${IXI_MAX_SUBJECTS:-120}"   # change here to keep more/less
IXI_SAMPLE_SEED="${IXI_SAMPLE_SEED:-2025}"

mkdir -p "$IXI_EVAL_DIR"

# Collect T1 files (flat or one-level nested under gen_IXI)
mapfile -t ixi_files < <(find "$GEN_IXI_DIR" -maxdepth 2 -type f \
  \( -name "*T1*.nii" -o -name "*T1*.nii.gz" \) | sort)

if (( ${#ixi_files[@]} > 0 )); then
  # Deterministic sampling by hashing full path
  pick_files=("${ixi_files[@]}")
  if (( IXI_MAX_SUBJECTS > 0 && ${#ixi_files[@]} > IXI_MAX_SUBJECTS )); then
    mapfile -t pick_files < <(
      for f in "${ixi_files[@]}"; do
        h=$(printf "%s" "${IXI_SAMPLE_SEED}:${f}" | sha256sum | awk '{print $1}')
        printf "%s\t%s\n" "$h" "$f"
      done | sort | awk -v n="$IXI_MAX_SUBJECTS" 'NR<=n{print}' | cut -f2-
    )
  fi

  # Copy chosen files directly into IXI_EVAL_DIR
  for f in "${pick_files[@]}"; do
    base=$(basename "$f")
    dst_file="$IXI_EVAL_DIR/$base"

    # overwrite any old junk and copy fresh
    rm -f -- "$dst_file"
    cp --reflink=auto "$f" "$dst_file" 2>/dev/null || cp "$f" "$dst_file"
  done

  say "Prepared IXI eval set with ${#pick_files[@]} files in $IXI_EVAL_DIR"
else
  echo "No IXI T1 NIfTI files detected under $GEN_IXI_DIR"
fi


# 9) Pretrained weights (optional)
if [[ -f best_model.pth ]]; then
  say "best_model.pth exists — skip"
else
  say "Downloading pretrained model…"
  if [[ "$USE_GDOWN" == "1" ]]; then
    gdown --id "$MODEL_ID" -O best_model.pth || true
  else
    wget -nv -O best_model.pth "https://drive.usercontent.google.com/download?id=${MODEL_ID}&export=download" || true
  fi
fi

# 10) Stats
say "OASIS train subjects: $(count_dirs "$OASIS_TRAIN_DIR")"
say "OASIS val subjects:   $(count_dirs "$OASIS_VAL_DIR")"
say "OASIS test subjects:  $(count_dirs "$OASIS_TEST_DIR")"
say "L2R gen files (img*.nii.gz): $(ls "$GEN_L2R_DIR"/img*.nii.gz 2>/dev/null | wc -l | tr -d ' ')"
say "IXI gen dir: $GEN_IXI_DIR (place or keep T1 NIfTIs here)"
