# LiquidReg
Normal mode: ./launch_LiquidReg_training_clr.sh
Debug mode (5% data): ./launch_LiquidReg_training_clr.sh --debug or ./launch_LiquidReg_training_clr.sh --debug-5pct
**One command to run everything:**

```bash
bash run_all.sh
```

That’s it. The script sets up the environment (conda must be installed) downloads and sets up data pairs, trains/validates on **OASIS**, and evaluates on **IXI** and **L2R**. Multi-GPU is handled automatically (DDP via `torchrun` if >1 GPU; otherwise plain Python).

---

## What it does

* **Data prep:** builds CSVs of registration pairs for:

  * `data/OASIS_train`, `data/OASIS_val`, `data/OASIS_test`
  * `data/gen_IXI_eval` (image-only metrics)
  * `data/gen_L2R` (labels if present)
* **Training:** pairs-CSV pipeline on OASIS (train/val).
* **Evaluation:** runs on OASIS test, IXI, and L2R using the trained checkpoint.
* **DDP:** classic data parallel. Gradients are synchronized; **only rank-0** writes TensorBoard and checkpoints.

---

## Outputs

```
runs/<experiment>/<seed>/
  checkpoints/
    best.pth         # single checkpoint (saved by rank-0)
  tensorboard/
    events...        # single TB stream (rank-0)
  eval/
    oasis_test/      # metrics.csv, summary.txt
    ixi/
    l2r/
```

---

## Data & pairs format

* CSV columns: `fixed, moving` and optionally `fixed_seg, moving_seg`.
* IXI typically has no segmentations; Dice/HD95 are computed only when labels exist.

---

## Key files (the ones that matter)

* `run_all.sh` — orchestrates data → train → eval (single command).
* `scripts/train.py` — training entrypoint (pairs-based loader, DDP-ready).
* `scripts/eval_pairs.py` — batch evaluator over a pairs CSV.
* `scripts/make_pairs.py` — creates deterministic pair lists.
* `dataloaders/reg_pairs_dataset.py` — **main** dataset/augment/patch loader.
* `models/` — `LiquidReg`, `LiquidRegLite`, transforms.
* `utils/` — preprocessing (normalize/resample), patch utils, misc helpers.
* `configs/` — configs and optional `cfgs_to_run.txt` list.

---

## Legacy (will be removed)

* `dataloaders/oasis_dataset.py` and any old “oasis\_\*” dataloader variants.
* Misc experimental/legacy scripts not referenced by `run_all.sh`.

> Only follow the chain started by `run_all.sh`. Other files are legacy or experimental.

---

## Minimal notes

* **Checkpoint to use:** `runs/<experiment>/<seed>/checkpoints/best.pth`
* **TensorBoard:** `runs/<experiment>/<seed>/tensorboard/ 
* **DDP behavior:** automatic with multiple GPUs; data folder is unchanged.
