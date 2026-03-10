# Sperm Morphology Classification — Deep Learning Case Study

**Author: Saurabh Mane** | © 2026 All Rights Reserved | [![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](LICENSE.md)

> ⚠️ **This is original work by Saurabh Mane.** Copying, reproducing, or submitting any part of this project — including code, results, analysis, or documentation — without explicit written permission is prohibited. See [LICENSE.md](LICENSE.md) for full terms.

---

> Three-class morphology classifier (normal / abnormal / non-sperm) built with ResNet50 and EfficientNet-B0, validated with calibration analysis, Grad-CAM, TTA sweep, and 5-fold cross-validation. Includes a REST API and web UI for inference.

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [Project Structure](#project-structure)
- [What was built](#what-was-built)
- [Results](#results)
- [Test-Time Augmentation](#test-time-augmentation)
- [Probability Calibration](#probability-calibration)
- [Grad-CAM Interpretability](#grad-cam-interpretability)
- [5-Fold Cross-Validation](#5-fold-cross-validation)
- [Model Comparison](#model-comparison)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Deployment Checklist](#deployment-checklist)
- [Known Limitations](#known-limitations)

---

## Why this project exists

Manual sperm morphology assessment is one of those tasks that looks straightforward on paper but gets messy in practice. Labs follow WHO criteria, but applying those criteria consistently across hundreds of cells per sample — across different analysts, different microscopes, different staining batches — produces a lot of inter-observer variability. The goal here was to build something that could work as a second opinion: not replace the andrologist, but flag cases worth a closer look and give confident, calibrated calls on the clear ones.

The first version used ResNet50 with focal loss and showed 87–88% accuracy. Good enough to publish, but not good enough to actually put in front of a clinical user. Before you can use model confidence scores for decision-making — routing borderline samples to human review, for instance — you need to know those scores actually mean something. That pushed the project into the extended validation work documented here.

---

## Project Structure

```
sperm-morphology/
├── src/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation and metrics
│   ├── preprocess.py         # CLAHE, background subtraction, edge enhancement
│   ├── split_data.py         # Stratified train/val/test split
│   └── visualize.py          # Feature maps and visualizations
├── experiments/
│   ├── gradcamexp.py         # Grad-CAM visualization for both models
│   ├── analyze_tta_effect.py # TTA sweep across N augmentations
│   ├── calibrate_resnet50.py # Temperature + Dirichlet calibration
│   ├── compare_models.py     # Head-to-head ResNet50 vs EfficientNet-B0
│   └── run_all_validations.py# 5-fold CV + full validation suite
├── api/
│   └── main.py               # FastAPI inference endpoint
├── models/
│   ├── best_resnet50.pth
│   └── best_efficientnet_b0.pth2
├── data/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── logs/
└── config_snapshot.yaml
```

---

## What was built

### Dataset

- ~3,000 microscopic sperm images across three classes: **normal**, **abnormal**, **non-sperm** (artifacts / debris)
- 70 / 15 / 15 stratified split → ~2,100 train / 450 val / 453 test images
- 224 × 224 px input, ImageNet mean/std normalization

The non-sperm class was included deliberately. Real lab samples always contain debris and artifacts, and a binary classifier that cannot reject them is not useful in production.

### Preprocessing

Three steps applied consistently at both train and inference time:

| Step | Purpose |
|------|---------|
| CLAHE | Boost local contrast in under-stained regions |
| Background subtraction | Isolate foreground cell structure |
| Edge enhancement | Sharpen head boundaries |

### Training Config

```yaml
training:
  model_type: efficientnet_b0   # or resnet50
  num_classes: 3
  num_epochs: 50
  learning_rate: 0.0001
  batch_size: 32
  loss_type: focal              # focal loss (gamma=2)
  dropout: 0.5
  scheduler: cosine
  use_class_weights: true
  use_mixed_precision: true
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
```

Both models use pretrained ImageNet weights with a replaced classification head: `GlobalAvgPool → Dropout(0.5) → Linear(3)`. Fine-tuned end-to-end from epoch 1.

---

## Results

### Classification Performance — ResNet50

| Class | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Normal | 0.82 | 0.85 | 0.837 | 0.947 |
| Abnormal | 0.93 | 0.90 | 0.917 | 0.991 |
| Non-sperm | 0.88 | 0.87 | 0.877 | 0.967 |
| **Macro avg** | **0.877** | **0.873** | **0.877** | **0.968** |
| **Accuracy** | | | **87.64%** | |

The abnormal class hits the highest F1 (0.917), which matters most clinically — missing an abnormal sample is the more consequential error in a fertility context.

---

## Test-Time Augmentation

TTA generates N augmented versions of each test image, runs them all through the model, then averages softmax outputs before argmax. Averaging over the augmentation-induced variance sharpens confident predictions and softens genuinely ambiguous ones.

We swept N ∈ {1, 3, 5, 7, 10}:

![TTA Comparison](tta_comparison.png)

| Metric | Without TTA | With TTA (N=5) |
|--------|-------------|----------------|
| Accuracy | 87.64% | **88.74%** |
| Macro F1 | 0.8771 | **0.8880** |

ResNet50 peaks at N=5 (+1.0 pp over baseline). EfficientNet-B0 peaks at N=7. **N=5 is the practical recommendation for ResNet50** — captures most of the gain at 5× inference cost, no meaningful improvement beyond that.

---

## Probability Calibration

Accuracy tells you how often the model is right. Calibration tells you whether the confidence scores are trustworthy. A model that says "90% confident" should be right 90% of the time — if it's only right 60% of the time at that threshold, the scores are useless for any downstream decision logic.

Both models came out of training noticeably overconfident. Two calibration methods were evaluated:

### Temperature Scaling

Fits a single scalar T on the validation set (both models converged to T ≈ 1.42). Simple, fast — and in this case, **it made things worse**:

| Model | ECE Before | ECE After Temp |
|-------|------------|----------------|
| ResNet50 | 0.120 | 0.178 ↑ |
| EfficientNet-B0 | 0.100 | 0.152 ↑ |

This happens when miscalibration is non-uniform across classes — one global scalar improves some confidence bins while hurting others. The reliability diagrams confirm it:

| ResNet50 | EfficientNet-B0 |
|----------|-----------------|
| ![ResNet50 before](ResNet50_calibration.png) | ![EfficientNet before](EfficientNet-B0_calibration.png) |
| Before calibration (ECE=0.120) | Before calibration (ECE=0.100) |
| ![ResNet50 temp](ResNet50_temp.png) | ![EfficientNet temp](EfficientNet-B0_temp.png) |
| After temperature scaling (ECE=0.178) | After temperature scaling (ECE=0.152) |

### Dirichlet Calibration

Fits a full linear transformation on log-probabilities — corrects class-wise bias and covariance structure simultaneously. Much more flexible than temperature scaling for problems where different classes have different confidence patterns.

| Model | ECE Before | ECE After Dirichlet |
|-------|------------|---------------------|
| ResNet50 | 0.120 | **0.036** ↓ |
| EfficientNet-B0 | 0.100 | **0.031** ↓ |

| ResNet50 | EfficientNet-B0 |
|----------|-----------------|
| ![ResNet50 dirichlet](ResNet50_dirichlet.png) | ![EfficientNet dirichlet](EfficientNet-B0_dirichlet.png) |
| Dirichlet calibrated (ECE=0.036) | Dirichlet calibrated (ECE=0.031) |

Both models now have ECE < 0.04. **For any deployment: fit Dirichlet calibration on your local validation set. The improvement is large and the fitting cost is negligible.**

---

## Grad-CAM Interpretability

Grad-CAM uses the gradient of the class score w.r.t. the last convolutional feature map to produce a spatial heatmap of what drove the prediction. The key question: is the model actually looking at the sperm head, or is it firing on background staining and slide artifacts?

### EfficientNet-B0 — Grad-CAM Samples (Set 1: samples 000–004)

Confidence scores: **55.73% → 56.62% → 87.44% → 89.56% → 72.99%**

| Sample | Image | Notes |
|--------|-------|-------|
| 000 — 55.73% | ![](sample_000___pred_normal_true_normal.png) | Multi-cell frame. Activation on primary head, spreads slightly — lower confidence from scene complexity, not wrong attention |
| 001 — 56.62% | ![](sample_001___pred_normal_true_normal.png) | Borderline head morphology. Heatmap is tight on nucleus — uncertainty is morphologically justified |
| 002 — 87.44% | ![](sample_002___pred_normal_true_normal.png) | Single sperm with tail. Clean head activation, tail completely ignored |
| 003 — 89.56% | ![](sample_003___pred_normal_true_normal.png) | Elongated sperm. Activation tracks head-midpiece junction precisely |
| 004 — 72.99% | ![](sample_004___pred_normal_true_normal.png) | Dual-head artifact in view. Model spreads activation across both heads, confidence drops accordingly |

### EfficientNet-B0 — Grad-CAM Samples (Set 2: samples 012–018)

This set is more diverse and includes one misclassification — the most instructive sample in the entire analysis.

| Sample | Image | Result | Notes |
|--------|-------|--------|-------|
| 012 — 59.45% | ![](sample_012___pred_normal_true_normal.png) | ✅ Correct | Sperm near slide edge — a difficult frame with the dark slide border visible. Model ignores the edge entirely and focuses on the head. Low confidence is appropriate given the partial occlusion |
| 013 — 82.82% | ![](sample_013___pred_normal_true_normal.png) | ✅ Correct | Clean elongated sperm, clear head. Compact, well-placed activation. One of the cleaner examples in the set |
| 014 — 76.53% | ![](sample_014___pred_normal_true_normal.png) | ✅ Correct | Multi-cell frame with debris in background. Model focuses tightly on the correct elongated sperm head in centre-right, ignoring the smaller debris cell on the left |
| **015 — 50.72%** | ![](sample_015___pred_non_sperm_true_normal.png) | ❌ **Misclassified** | **True: normal — Pred: non_sperm.** The model sees a large pear-shaped head with an unusually elongated midpiece attachment, and at 50.72% confidence it barely tips over the non_sperm threshold. The Grad-CAM activation covers the full head-neck region rather than just the nucleus, which is consistent with the model being genuinely confused about what structure it is looking at. This is the most borderline morphology in the set |
| 016 — 77.57% | ![](sample_016___pred_normal_true_normal.png) | ✅ Correct | Single sperm with tail fully in frame. Activation precisely on head, tail ignored |
| 017 — 64.56% | ![](sample_017___pred_normal_true_normal.png) | ✅ Correct | Abnormal-looking tail attachment — head activation is correct but confidence is lower, which makes sense given the unusual midpiece morphology |
| 018 — 74.51% | ![](sample_018___pred_normal_true_normal.png) | ✅ Correct | **Hardest correct prediction in the set.** Very cluttered background with debris and overlapping cellular material everywhere. Despite this, the model locks onto the sperm head in centre-frame. This is a strong result — the preprocessing pipeline (CLAHE + background subtraction) is almost certainly doing useful work here |

### ResNet50 — Grad-CAM Samples (Set 1: samples 000–004)

Same frames as EfficientNet Set 1. ResNet50 is ~20 pp more confident but CV accuracy (79.19%) shows this confidence is not backed by better generalisation.

| Sample | Image | Notes |
|--------|-------|-------|
| 000 — 84.08% | ![](sample_000___pred_normal_true_normal.png) | Same multi-cell frame — commits to one head, 28 pp more confident than EfficientNet |
| 001 — 91.04% | ![](sample_001___pred_normal_true_normal.png) | Highest confidence in set. Tight circular activation on acrosomal region |
| 002 — 88.53% | ![](sample_002___pred_normal_true_normal.png) | Head+midpiece activation. Both models agree on this frame |
| 003 — 86.93% | ![](sample_003___pred_normal_true_normal.png) | Elongated sperm. Head activation, no tail bleed |
| 004 — 89.14% | ![](sample_004___pred_normal_true_normal.png) | Dual-head frame — focuses on larger head, 16 pp more confident than EfficientNet |

### ResNet50 — Grad-CAM Samples (Set 2: samples 015–019)

This set exposes ResNet50's failure mode directly — including a misclassification **on the exact same frame** that EfficientNet-B0 got right.

| Sample | Image | Result | Notes |
|--------|-------|--------|-------|
| 015 — 62.94% | ![](resnet_sample_015___pred_normal_true_normal.png) | ✅ Correct | Same pear-shaped sperm that EfficientNet misclassified as non_sperm at 50.72%. ResNet50 gets it right here, but at only 62.94% — both models are uncertain, they just fail on different edge cases |
| 016 — 88.20% | ![](resnet_sample_016___pred_normal_true_normal.png) | ✅ Correct | Single elongated sperm with tail. Clean head-midpiece activation, confident and correct |
| 017 — 84.49% | ![](resnet_sample_017___pred_normal_true_normal.png) | ✅ Correct | Unusual tail attachment at head-midpiece junction. Correct head focus, both models handle this well |
| **018 — 70.34%** | ![](resnet_sample_018___pred_abnormal_true_normal.png) | ❌ **Misclassified** | **True: normal — Pred: abnormal (70.34%).** This is the exact same heavily cluttered background frame where EfficientNet-B0 predicted normal correctly at 74.51%. ResNet50 misclassifies it as abnormal — the dense debris and overlapping cellular material confuse it into seeing morphological defects that aren't there. The Grad-CAM lands on the right region but the global scene context overwhelms the classification. This is a direct same-frame head-to-head where EfficientNet wins |
| 019 — 74.29% | ![](resnet_sample_019___pred_normal_true_normal.png) | ✅ Correct | Small triangular head. Lower confidence appropriate for this compact, somewhat atypical morphology |

### Key takeaways

- **Sample_018 is the single most important finding across all Grad-CAM results.** It is the same cluttered frame, same model architecture family, same preprocessing — EfficientNet-B0 correct at 74.51%, ResNet50 wrong at 70.34% abnormal. This is not a statistical abstraction — it is a direct demonstration of why EfficientNet generalises better on hard cases
- **Both models' misclassifications fall below the 0.80 confidence threshold:** EfficientNet's non_sperm error at 50.72% and ResNet50's abnormal error at 70.34% would both route to human review in a properly deployed system — the calibration and threshold logic is working
- EfficientNet misclassifies a morphologically unusual head (atypical shape → non_sperm). ResNet50 misclassifies a visually noisy background (clutter → abnormal). The ResNet50 failure mode is more clinically concerning — it is being distracted by the slide environment rather than the cell itself
- ResNet50 looks more confident and more spatially focused on easy, clean frames — but that confidence collapses on harder real-world conditions. EfficientNet's lower but more calibrated confidence better reflects actual difficulty
- **Every failure case here would be caught at 0.80 threshold → human review.** The combination of Dirichlet calibration + confidence threshold is the correct deployment pattern for both models

---

## 5-Fold Cross-Validation

With ~3,000 images, a single split is noisy. 5-fold stratified CV (80/20, same augmentation config) gives a more reliable performance estimate and quantifies variance across partitions.

| Fold | EfficientNet-B0 Acc | EffNet F1-Abnormal | EffNet ROC-AUC | ResNet50 Acc | ResNet F1-Abnormal | ResNet ROC-AUC |
|------|--------------------|--------------------|----------------|-------------|-------------------|----------------|
| 1 | 85.49% | 0.931 | 0.9536 | 81.37% | 0.885 | 0.9414 |
| 2 | 84.12% | 0.904 | 0.9492 | 77.84% | 0.854 | 0.9340 |
| 3 | 82.91% | 0.921 | 0.9573 | 83.10% | 0.862 | 0.9533 |
| 4 | 84.87% | 0.914 | 0.9624 | 74.85% | 0.761 | 0.9145 |
| 5 | **88.02%** | **0.933** | **0.9653** | 78.78% | 0.857 | 0.9420 |
| **Mean ± SD** | **85.08 ± 1.90%** | **0.921 ± 0.012** | **0.9576 ± 0.0065** | **79.19 ± 3.20%** | **0.844 ± 0.048** | **0.9370 ± 0.0144** |

EfficientNet-B0 wins every fold on accuracy except fold 3 (where it loses by 0.2 pp). More importantly, ResNet50 shows much higher variance — fold 4 loss hits 1.006 vs EfficientNet's 0.489, and its accuracy SD (3.20%) is nearly double EfficientNet's (1.90%). That instability is a real concern for a model you want to trust in production.

---

## Model Comparison

| Criterion | EfficientNet-B0 | ResNet50 |
|-----------|-----------------|----------|
| CV mean accuracy | **85.08 ± 1.90%** ✅ | 79.19 ± 3.20% |
| CV F1-abnormal (mean) | **0.921 ± 0.012** ✅ | 0.844 ± 0.048 |
| CV ROC-AUC (mean) | **0.9576 ± 0.0065** ✅ | 0.9370 ± 0.0144 |
| CV loss stability (SD) | **0.096** ✅ | 0.216 |
| ECE (Dirichlet) | **0.031** ✅ | 0.036 |
| Model size | **~20 MB** ✅ | ~98 MB |
| Inference (CPU) | **~180ms/img** ✅ | ~420ms/img |
| Optimal TTA | N=7 | N=5 |
| **Overall verdict** | **✅ Deploy this** | ❌ Loses on CV |

> **EfficientNet-B0 wins on every meaningful metric.** More accurate, more stable across folds, better calibrated, 5× smaller, and 2.3× faster. There is no trade-off here.

---

## Running the Project

### 1. Data preparation

```bash
# Preprocess raw images (CLAHE + background subtraction + edge enhancement)
python -m src.preprocess

# Create stratified train / val / test splits
python -m src.split_data
```

### 2. Training

```bash
python -m src.train
```

### 3. Evaluation and visualization

```bash
# Metrics on test set
python -m src.evaluate

# Feature map and activation visualizations
python -m src.visualize
```

### 4. Experiments

```bash
# Grad-CAM — EfficientNet-B0
python -m experiments.gradcamexp \
  --model efficientnet_b0 \
  --weights models/best_efficientnet_b0.pth2

# Grad-CAM — ResNet50
python -m experiments.gradcamexp \
  --model resnet50 \
  --weights models/best_resnet50.pth

# TTA sweep (N = 1, 3, 5, 7, 10)
python -m experiments.analyze_tta_effect

# Calibration — temperature scaling + Dirichlet (ResNet50)
python -m experiments.calibrate_resnet50

# Head-to-head model comparison
python -m experiments.compare_models

# Full validation suite — 5-fold CV + all metrics
python -m experiments.run_all_validations
```

### 5. API server

```bash
uvicorn api.main:app --reload
```

Runs at `http://127.0.0.1:8000` — Swagger docs at `http://127.0.0.1:8000/docs`

---

## API Reference

### `POST /predict`

Classifies a single sperm image and returns calibrated probabilities.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.png"
```

**Response**
```json
{
  "prediction": "normal",
  "confidence": 0.891,
  "probabilities": {
    "normal": 0.891,
    "abnormal": 0.082,
    "non_sperm": 0.027
  }
}
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "model": "resnet50" }
```

### `GET /docs`

Interactive Swagger UI — open `http://127.0.0.1:8000/docs` in browser.

---

## Deployment Checklist

- [ ] **Detection stage** — add a cell detector (YOLO or similar) to extract single-cell crops from raw microscope frames before passing to the classifier
- [ ] **Dirichlet calibration** — fit on local validation data from the target lab's microscope; don't reuse calibration weights from a different source
- [ ] **Confidence threshold** — ~0.80 for auto-classification; route below-threshold to human review
- [ ] **Domain validation** — test on the target lab's microscope and staining protocol before go-live
- [ ] **Motility + concentration** — morphology is one semen parameter; a complete fertility assessment needs additional modules

---

## Known Limitations

**Single-source data.** All images from one lab with one staining protocol. Generalization to a different microscope or staining batch is unvalidated — probably the biggest real-world risk.

**Multi-cell frames.** The model focuses on whichever sperm head is most salient in the feature map, not a deliberate cell selection. A detection stage fixes this.

**Borderline morphology.** ECE is highest in the 0.4–0.6 confidence range, where normal/abnormal boundary cases land. This is inherent to the problem — experienced andrologists disagree on these cases too.

**Morphology only.** This is one piece of a complete semen analysis, not a standalone diagnostic tool.

---

## What's next

- Multi-center dataset expansion for generalization across labs and devices
- Detection + classification pipeline (YOLO detector → single-cell classifier)
- Motility analysis module (optical flow / CASA-style tracking)
- Prospective clinical validation study

---

*Built at Vidyavardhini's College of Engineering and Technology, Dept. of AI & DS.*
