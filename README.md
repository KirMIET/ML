## Model Development

The final model was developed using **two parallel approaches**.

### Approach 1: Powerful Ensemble with Advanced Techniques

In the first approach, a high-performance ensemble was built consisting of **25 neural networks** (5 architectures × 5 folds).

**Key techniques used:**

- **Model architectures** (via `timm` library):
  - `regnety_040`
  - `maxvit_tiny_tf_224`
  - `swin_tiny_patch4_window7_224`
  - `deit3_small_patch16_224`
  - `tf_efficientnetv2_s`

- **5-Fold Stratified Cross-Validation** with subclass-aware splitting
- **Heavy augmentations** (Albumentations): ShiftScaleRotate, RandomRotate90, Horizontal & Vertical Flip, GaussianBlur, GaussNoise, CoarseDropout
- **MixUp + Label Smoothing**
- **Custom loss**: `SoftFocalLoss` (γ=2.0) with class weights
- **Optimizer**: AdamW with Cosine Annealing LR scheduler
- **Regularization**: DropPath, Gradient Clipping, Automatic Mixed Precision (AMP)
- **Test-Time Augmentation** (4 variants: original + hflip + vflip + hvflip)
- **Ensemble aggregation** using **Power Weighting** (p=4.0) based on per-fold F1-score

### Approach 2: [To be described]

---

*Details about the second approach will be added later.*