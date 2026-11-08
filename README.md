# Distilling the Night: Narrowing the Day-to-Night Gap of Lightweight YOLOv8 with Knowledge Distillation

## Abstract

Lightweight object detectors deployed on in-vehicle edge devices need to work under both daylight and darkness with a single model, but small models degrade sharply at night. This project distills a fine-tuned **YOLOv8-L** teacher (43.6M params) into a **YOLOv8-S** student (11.1M params) on a day/night-balanced subset of **BDD100K**, evaluating each model separately on daytime and nighttime test sets. The distillation framework combines multi-level feature alignment (neck layers P3–P5) with response distillation (classification logits + DFL box distributions), plus three mechanisms aimed at nighttime failure modes:

- **Soft confidence weighting** — weights each anchor continuously by teacher confidence instead of a hard foreground mask
- **Cosine-decayed distillation weight** — anneals teacher supervision from 0.35 → 0.10 over training
- **Channel-wise feature distillation (CWD)** — matches spatial attention patterns rather than raw activation magnitudes, which are lighting-dependent

**Key result:** averaged over 3 runs of each student, the distilled student's nighttime mAP@0.5 rises from 0.4571 to 0.4780 (+0.0209) while daytime mAP@0.5 also improves (+0.0104), shrinking the relative day-to-night drop from 6.90% (baseline) to 4.65%, below the teacher's 7.88%, with no added inference cost. Under 5-fold cross-validation the distilled student beats the baseline in all five folds under both conditions, while the reduction in relative drop (5.77% → 5.00%) varies across folds. The largest nighttime gains are for vulnerable road users (rider +0.0625, motor +0.0366, bike +0.0356 AP@0.5).

## Results Summary

**Fixed split** (2,000 day + 2,000 night test images; students = mean of 3 runs, teacher = 1 run)

| Model | mAP@0.5 Day | mAP@0.5 Night | Rel. drop |
| --- | --- | --- | --- |
| Teacher YOLOv8-L | 0.5597 | 0.5156 | 7.88% |
| Student baseline YOLOv8-S | 0.4909 ± 0.0014 | 0.4571 ± 0.0096 | 6.90% ± 1.71 |
| Student KD YOLOv8-S | 0.5013 ± 0.0024 | 0.4780 ± 0.0038 | 4.65% ± 1.05 |

**5-fold cross-validation** (same folds for baseline and KD)

| Model | mAP@0.5 Day | mAP@0.5 Night | Rel. drop |
| --- | --- | --- | --- |
| Student baseline | 0.4990 ± 0.0060 | 0.4703 ± 0.0128 | 5.77% ± 1.72 |
| Student KD | 0.5145 ± 0.0074 | 0.4887 ± 0.0075 | 5.00% ± 1.89 |

**Inference** (NVIDIA T4, batch 1, 640 × 640): teacher 43.6M params / 165.4 GFLOPs / 42.7 ms FP32; student 11.1M params / 28.7 GFLOPs / 10.1–10.2 ms FP32. Student KD and the baseline are identical in size and within one SD in latency.

Relative drop = (mAP@0.5 day − mAP@0.5 night) / mAP@0.5 day × 100.

## Repository Structure

This repo contains the Jupyter notebooks, training logs, and evaluation outputs used to produce the paper's results. All models are YOLOv8, trained/fine-tuned on a 9-class subset of BDD100K (the `train` class was dropped for being statistically negligible).

```
teacher/                     Fine-tune YOLOv8-L teacher (COCO-pretrained -> 9-class head) + evaluation
student baseline 3x run/     YOLOv8-S student without teacher guidance, trained 3x (comparison baseline)
  1 - run 1 / 2 - run 2 / 3 - run 3
student kd 3x run/           Distilled YOLOv8-S student, trained 3x under non-deterministic CUDA
  1 - run 1 / 2 - run 2 / 3 - run 3
ablation study/              Leave-one-out ablation: disable one KD mechanism at a time
  1 - no soft weighting      Hard foreground mask instead of soft confidence weighting
  2 - no beta decay          Constant distillation weight (beta = 0.225) instead of cosine schedule
  3 - no cwd                 Standard MSE feature loss instead of channel-wise distillation
kfold/                       Stratified 5-fold cross-validation (baseline and KD on identical folds)
  make_folds/                Builds folds.json (stratified by time-of-day + per-class counts)
  student baseline/          base-kfold-N-{train,resume,eval}.ipynb + per-fold CSVs, aggregate/
  student kd/                kfold-N-{training,resume,eval}.ipynb + per-fold CSVs, aggregate/
  aggregate/                 Paired baseline-vs-KD comparison per fold (paired t-tests)
inference benchmark/         Params, GFLOPs, model size, VRAM, and FP32/FP16 latency on a Kaggle T4
```

Each model directory typically contains:

- training / fine-tuning notebook (`*-training.ipynb`, `*-train.ipynb`, `*-ft.ipynb`, `notebook2-*`, `notebook3-kd*`)
- `*-resume*.ipynb` — resume-from-checkpoint notebook (Kaggle sessions have a time limit, so long runs were resumed from the best checkpoint)
- `*-eval*.ipynb` — evaluation notebook on the daytime and nighttime test splits
- `output-eval/` and `training-history/` (spelled `output eval/` / `training history/` in some baseline runs) — resulting CSV metrics (overall and per-class mAP/Precision/Recall)

Notes:

- The aggregates report relative drop in different units: `kfold/student kd/aggregate/kfold_summary.csv` uses a fraction (0.0500), while the two newer aggregates use percent (5.00).
- The per-fold CSVs for KD fold 2 are not in the repo; its numbers are included in the aggregate CSVs.
- The inference benchmark used the best checkpoint of KD run 3.

## Dataset

[BDD100K](https://bdd-data.berkeley.edu/), filtered to daytime and nighttime images only, with 28,000 total images (14,000 day / 14,000 night) split into training (20,000), validation (4,000), and testing (4,000) — balanced 50/50 between conditions in every split so any day/night performance gap can be attributed to lighting rather than data volume. The 5-fold cross-validation draws its test folds (about 5,530 images each) from the 27,669 images of the official validation and test directories.

## Training Configuration (summary)

| Parameter                                   | Value                               |
| ------------------------------------------- | ----------------------------------- |
| Input resolution                            | 640 × 640                           |
| Optimizer                                   | AdamW, weight decay 0.05            |
| Epochs (teacher FT / student baseline / KD) | 30 / 50 / 50                        |
| Early stopping                              | Patience 14 on mean validation loss |
| Task loss weight α                          | 0.7                                 |
| Response distillation weight β(t)           | Cosine decay 0.35 → 0.10            |
| Feature distillation weight γ               | 0.03                                |
| Classification / DFL / CWD temperature      | 4.0 / 2.0 / 4.0                     |
| Feature levels                              | Neck layers 15, 18, 21 (P3, P4, P5) |

## Pipeline Overview (not a runnable checkout)

1. Fine-tune the YOLOv8-L teacher (`teacher/`) and freeze the best checkpoint.
2. Train the YOLOv8-S baseline with no teacher guidance, 3x (`student baseline 3x run/`).
3. Distill the student using the frozen teacher, 3x (`student kd 3x run/`).
4. Disable each KD mechanism in turn for the leave-one-out ablation (`ablation study/`).
5. Build the 5 folds, train/eval the baseline and KD student on each fold, then aggregate (`kfold/`).
6. Benchmark size and latency of the three fixed-split models (`inference benchmark/`).

> **⚠️ Disclaimer:** All training and evaluation was originally run on **Kaggle notebooks**, split across each author's own Kaggle account (for GPU quota reasons). The notebooks in this repo were collected afterward into a single place for documentation/archival purposes — they are **not plug-and-play runnable as-is**. Expect Kaggle-specific dataset mount paths (`/kaggle/input/...`), per-account working directories, and possibly missing dataset/checkpoint files that lived on the original Kaggle accounts. Treat this repo as a record of the exact code and results behind the paper, not a one-command reproduction pipeline — re-running it will likely need path fixes and re-uploading the BDD100K subset/checkpoints to your own environment.
