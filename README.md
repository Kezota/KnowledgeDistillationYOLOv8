# Enhancing Daytime-to-Nighttime Robustness of YOLOv8 Object Detection Using Knowledge Distillation

Undergraduate thesis project (BINUS University, Computer Science).

**Authors:** Yoseph Oktavianus Yusanto, William Hartanto, Kezia Meilany Tandapai (supervised by Anderies)

## Abstract

Lightweight object detectors deployed on in-vehicle edge devices need to work under both daylight and darkness with a single model, but small models degrade sharply at night. This project distills a fine-tuned **YOLOv8-L** teacher (43.7M params) into a **YOLOv8-S** student (11.2M params) on a day/night-balanced subset of **BDD100K**, evaluating each model separately on daytime and nighttime test sets. The distillation framework combines multi-level feature alignment (neck layers P3–P5) with response distillation (classification logits + DFL box distributions), plus three mechanisms aimed at nighttime failure modes:

- **Soft confidence weighting** — weights each anchor continuously by teacher confidence instead of a hard foreground mask
- **Cosine-decayed distillation weight** — anneals teacher supervision from 0.35 → 0.10 over training
- **Channel-wise feature distillation (CWD)** — matches spatial attention patterns rather than raw activation magnitudes, which are lighting-dependent

**Key result:** averaged over 3 runs, the distilled student's nighttime mAP@0.5 rises from 0.4470 to 0.4780 (+0.0310) while daytime mAP@0.5 also improves (+0.0120), shrinking the relative day-to-night performance drop from 8.65% (baseline) to 4.65% — below the teacher's own 7.88% drop — with no added inference cost. Gains hold across a leave-one-out ablation and 5-fold cross-validation, and are largest for vulnerable road users at night (bike +0.1082, rider +0.1055 AP@0.5).

## Repository Structure

This repo contains the Jupyter notebooks, training logs, and evaluation outputs used to produce the paper's results. All models are YOLOv8, trained/fine-tuned on a 9-class subset of BDD100K (the `train` class was dropped for being statistically negligible).

```
teacher/                 Fine-tune YOLOv8-L teacher (COCO-pretrained -> 9-class head) + evaluation
student baseline/        Train YOLOv8-S student from scratch, no teacher guidance (comparison baseline)
student kd 3x run/       Distilled YOLOv8-S student, trained 3x under non-deterministic CUDA
  1 - run 1 / 2 - run 2 / 3 - run 3
ablation study/          Leave-one-out ablation: disable one KD mechanism at a time
  1 - no soft weighting  Hard foreground mask instead of soft confidence weighting
  2 - no beta decay      Constant distillation weight (beta = 0.225) instead of cosine schedule
  3 - no cwd             Standard MSE feature loss instead of channel-wise distillation
kfold/                   Stratified 5-fold cross-validation of the distilled student
  make_folds/            Builds the 5 folds (stratified by time-of-day + per-class counts)
  0 - fold = 0 ... 4 - fold = 4   Train + eval per fold
  aggregate/              Aggregates fold results into the mean/SD reported in the paper
```

Each model directory typically contains:

- `*-training.ipynb` / `*-ft.ipynb` — training or fine-tuning notebook
- `*-resume*.ipynb` — resume-from-checkpoint notebook (non-deterministic runs sometimes needed a restart)
- `*-eval*.ipynb` — evaluation notebook, run separately on the daytime and nighttime test splits
- `output-eval/` and `training-history/` — resulting CSV metrics (overall and per-class mAP/Precision/Recall)

## Dataset

[BDD100K](https://bdd-data.berkeley.edu/), filtered to daytime and nighttime images only, with 28,000 total images (12,000 day / 12,000 night) split into training (20,000), validation (4,000), and testing (4,000) — balanced 50/50 between conditions in every split so any day/night performance gap can be attributed to lighting rather than data volume.

## Training Configuration (summary)

| Parameter                                   | Value                               |
| ------------------------------------------- | ----------------------------------- |
| Input resolution                            | 640 × 640                           |
| Optimizer                                   | AdamW, weight decay 0.05            |
| Epochs (teacher FT / student baseline / KD) | 30 / 50 / 50                        |
| Task loss weight α                          | 0.7                                 |
| Response distillation weight β(t)           | Cosine decay 0.35 → 0.10            |
| Feature distillation weight γ               | 0.03                                |
| Classification / DFL / CWD temperature      | 4.0 / 2.0 / 4.0                     |
| Feature levels                              | Neck layers 15, 18, 21 (P3, P4, P5) |

Full details are in the paper (`Skripsi_Final.pdf`).

## Pipeline Overview (not a runnable checkout)

1. Fine-tune the YOLOv8-L teacher (`teacher/`) and freeze the best checkpoint.
2. Train the YOLOv8-S baseline with no teacher guidance (`student baseline/`).
3. Distill the student using the frozen teacher (`student kd 3x run/`, repeated 3x for the reported mean ± SD).
4. Disable each KD mechanism in turn for the leave-one-out ablation (`ablation study/`).
5. Build the 5 folds and train/eval per fold for cross-validation (`kfold/`).

> **⚠️ Disclaimer:** All training and evaluation was originally run on **Kaggle notebooks**, split across each author's own Kaggle account (for GPU quota reasons). The notebooks in this repo were collected afterward into a single place for documentation/archival purposes — they are **not plug-and-play runnable as-is**. Expect Kaggle-specific dataset mount paths (`/kaggle/input/...`), per-account working directories, and possibly missing dataset/checkpoint files that lived on the original Kaggle accounts. Treat this repo as a record of the exact code and results behind the paper, not a one-command reproduction pipeline — re-running it will likely need path fixes and re-uploading the BDD100K subset/checkpoints to your own environment.
