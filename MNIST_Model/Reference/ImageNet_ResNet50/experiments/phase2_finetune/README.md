# Phase 2: Fine-Tuning with Advanced Augmentation

## Overview
Phase 2 builds upon Phase 1 by adding advanced regularization and augmentation techniques to improve model generalization and reduce overfitting.

## Improvements Over Phase 1

### 1. Dropout Regularization
- **Dropout Rate**: 0.02
- Applied in both convolutional blocks and the final classifier
- Helps prevent overfitting and improves generalization

### 2. Mixup Augmentation
- **Alpha**: 0.2
- Blends pairs of training images and their labels
- Formula: `mixed_image = λ * image_a + (1 - λ) * image_b`
- Encourages the model to behave linearly between training examples

### 3. Cutmix Augmentation
- **Alpha**: 1.0
- Cuts and pastes patches between training images
- Maintains spatial information better than Mixup
- More effective for vision tasks

### 4. RandAugment
- **N**: 2 (number of augmentation operations)
- **M**: 9 (magnitude of augmentations)
- Includes transformations like:
  - Geometric: Rotation, Shift, Scale, Elastic Transform
  - Noise: Gaussian Noise, Blur, Motion Blur
  - Color: Brightness/Contrast, Hue/Saturation, RGB Shift

### 5. Progressive Resizing
- **Input Size**: 224×224 (increased from 128×128 in Phase 1)
- Full ImageNet resolution for better feature learning

### 6. Optimizer Changes
- **Optimizer**: AdamW (better for fine-tuning)
- **Learning Rate**: 0.0001 (lower for fine-tuning)
- **Scheduler**: CosineAnnealingLR (smooth decay)

## Configuration

```python
# Model
dropout_rate: 0.02
pretrained: Phase 1 best model

# Data Augmentation
use_randaugment: True
randaugment_n: 2
randaugment_m: 9
use_mixup: True
mixup_alpha: 0.2
use_cutmix: True
cutmix_alpha: 1.0

# Training
epochs: 50
learning_rate: 0.0001
optimizer: AdamW
scheduler: CosineAnnealingLR
input_size: 224
batch_size: 32
gradient_accumulation: 4 (effective batch = 128)
```

## Expected Results

### Target Metrics
- **Top-1 Accuracy**: 65-70%
- **Top-5 Accuracy**: 85-90%
- **Improvement over Phase 1**: +5-10% top-1 accuracy
- **Better generalization**: Smaller train-val accuracy gap

### Training Characteristics
- Slower initial convergence (due to augmentation)
- Lower final training accuracy (due to regularization)
- Higher validation accuracy (better generalization)
- More stable training curves

## Running Phase 2

### Prerequisites
1. Complete Phase 1 training successfully
2. Have Phase 1 best model at `../phase1_subset_20pct/models/best_model.pth`
3. ImageNet dataset prepared and available

### Execute Training

```bash
# Navigate to phase2 directory
cd ImageNet_ResNet50/experiments/phase2_finetune

# Run training
python train.py
```

### Monitor Progress

Training outputs will be saved to:
- **Logs**: `logs/imagenet_training.log`
- **Models**: `models/best_model.pth`, `models/checkpoint_epoch_*.pth`
- **Metrics**: `logs/training_metrics.csv`
- **Curves**: `logs/training_curves_*.png`

## Key Implementation Details

### Mixup/Cutmix Application
```python
# Applied during training only
if mixup_cutmix is enabled:
    images, targets_a, targets_b, lam = mixup_cutmix(images, targets)
    loss = lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)
```

### Dropout in Model
```python
# Applied in BottleneckBlocks and final classifier
self.dropout = nn.Dropout2d(dropout_rate)  # In conv blocks
self.dropout = nn.Dropout(dropout_rate)    # Before FC layer
```

### RandAugment Pipeline
```python
# Multiple augmentation groups applied randomly
transforms = [
    A.OneOf([ShiftScaleRotate, ElasticTransform, GridDistortion]),
    A.OneOf([GaussNoise, GaussianBlur, MotionBlur]),
    A.OneOf([RandomBrightnessContrast, HueSaturationValue, RGBShift])
]
```

## Troubleshooting

### Phase 1 Model Not Found
If you get an error about missing Phase 1 model:
```bash
# Option 1: Train Phase 1 first
cd ../phase1_subset_20pct
python train.py

# Option 2: Train from scratch (slower, lower accuracy)
# The script will automatically fallback to training from scratch
```

### Out of Memory (OOM)
If you run out of GPU memory:
```python
# Reduce batch size in train.py
config.data.batch_size = 16  # Instead of 32

# Or reduce input size (not recommended)
config.data.input_size = 128  # Instead of 224
```

### Slow Training
Augmentation makes training slower:
- Mixup/Cutmix: ~10% slower
- RandAugment: ~15% slower
- Combined: ~25% slower than Phase 1

This is expected and worthwhile for better generalization.

## Validation

### Compare with Phase 1
```python
# Expected improvements:
- Phase 1 Top-1: 50-60%
- Phase 2 Top-1: 65-70%
- Generalization gap: Reduced by 5-10%
```

### Check for Overfitting
Monitor the training curves:
- Train-val accuracy gap should be smaller than Phase 1
- Validation loss should be more stable
- No early overfitting in initial epochs

## Next Steps

After Phase 2 completion:

1. **Analyze Results**
   - Compare metrics with Phase 1
   - Review training curves
   - Check confusion matrix

2. **Grad-CAM Analysis**
   - Identify remaining misclassifications
   - Understand model attention patterns
   - Find classes that need more work

3. **Phase 3 Planning**
   - Prepare for full dataset training
   - Consider ensemble methods
   - Plan production deployment

## Files Structure

```
phase2_finetune/
├── train.py              # Main training script
├── README.md            # This file
├── logs/                # Training logs and metrics
│   ├── imagenet_training.log
│   ├── training_metrics.csv
│   └── training_curves_*.png
└── models/              # Model checkpoints
    ├── best_model.pth
    └── checkpoint_epoch_*.pth
```

## References

- **Mixup**: [Zhang et al., 2017](https://arxiv.org/abs/1710.09412)
- **Cutmix**: [Yun et al., 2019](https://arxiv.org/abs/1905.04899)
- **RandAugment**: [Cubuk et al., 2020](https://arxiv.org/abs/1909.13719)
- **Dropout**: [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)

---

**Author**: Krishnakanth  
**Date**: October 16, 2025  
**Status**: Ready for training

