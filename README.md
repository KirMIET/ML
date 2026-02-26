## Model Development

The final solution was developed using **two parallel approaches**.

### Approach 1: Powerful 25-Model Ensemble

In the first approach, a high-performance ensemble was built consisting of **25 neural networks** (5 different architectures × 5 folds).

**Key techniques used:**

- **Architectures** (via `timm`):
  - `regnety_040`
  - `maxvit_tiny_tf_224`
  - `swin_tiny_patch4_window7_224`
  - `deit3_small_patch16_224`
  - `tf_efficientnetv2_s`

- 5-Fold Stratified Cross-Validation with subclass-aware splitting
- Heavy augmentations (Albumentations): ShiftScaleRotate, RandomRotate90, flips, GaussianBlur, GaussNoise, CoarseDropout
- MixUp + Label Smoothing
- Custom loss: `SoftFocalLoss` (γ=2.0) with class weights
- AdamW optimizer + Cosine Annealing LR
- Gradient clipping, DropPath, Automatic Mixed Precision (AMP)
- Test-Time Augmentation (4 variants)
- Power Weighting (p=4.0) ensemble aggregation based on per-fold F1-score

### Approach 2: Single Strong EfficientNetV2 Model (6-Fold)

In the second approach, a single powerful model — **`tf_efficientnetv2_s.in21k_ft_in1k`** (pretrained on ImageNet-21k and fine-tuned on ImageNet-1k) — was trained using **6-fold cross-validation**.

**Key features:**

- Input resolution increased to **300×300**
- Strong augmentations including ColorJitter, CLAHE, and advanced color distortions
- MixUp (α=0.4) + Label Smoothing
- Loss: `SoftTargetCrossEntropy`
- AdamW optimizer with Cosine Annealing LR scheduler
- 6 models trained (one per fold)
- Test-Time Augmentation (original + hflip + vflip + hvflip)
- Weighted averaging of predictions using F1-scores from each fold

---
