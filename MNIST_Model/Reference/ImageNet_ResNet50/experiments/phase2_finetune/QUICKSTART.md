# Phase 2 Quick Start Guide

## Prerequisites

✅ Phase 1 training completed  
✅ Phase 1 best model available at: `../phase1_subset_20pct/models/best_model.pth`  
✅ ImageNet dataset downloaded and extracted  
✅ GPU with 12+ GB VRAM (recommended)

## Run Phase 2 Training

```bash
# 1. Navigate to phase2 directory
cd ImageNet_ResNet50/experiments/phase2_finetune

# 2. Activate your PyTorch environment
source /opt/pytorch/bin/activate  # On EC2
# OR
conda activate pytorch_2_8         # On conda

# 3. Run training
python train.py
```

That's it! The script will:
- Load Phase 1 model automatically
- Apply dropout (0.02)
- Enable Mixup, Cutmix, and RandAugment
- Train for 50 epochs
- Save best model to `models/best_model.pth`

## Expected Runtime

- **Per Epoch**: ~45-60 minutes (on subset)
- **Total Training**: ~40-50 hours
- **Speed**: ~25% slower than Phase 1 (due to augmentation)

## Monitor Progress

### View Logs
```bash
# Real-time log monitoring
tail -f logs/imagenet_training.log
```

### Check Metrics
```bash
# View training metrics CSV
cat logs/training_metrics.csv

# Or open with Excel/pandas
```

### View Training Curves
Training curves are automatically saved after completion:
```
logs/training_curves_YYYYMMDD_HHMMSS.png
```

## Expected Results

### Target Metrics
- **Top-1 Accuracy**: 65-70%
- **Top-5 Accuracy**: 85-90%
- **Train-Val Gap**: < 10% (improved from Phase 1's 15-20%)

### What Good Training Looks Like
```
Epoch 1:  Loss: 5.243, Top-1: 15.23%, Top-5: 35.67%
Epoch 10: Loss: 3.156, Top-1: 35.78%, Top-5: 62.45%
Epoch 20: Loss: 2.234, Top-1: 52.34%, Top-5: 78.23%
Epoch 30: Loss: 1.876, Top-1: 61.45%, Top-5: 84.67%
Epoch 40: Loss: 1.654, Top-1: 66.23%, Top-5: 87.89%
Epoch 50: Loss: 1.523, Top-1: 68.45%, Top-5: 89.34%
```

## Troubleshooting

### Phase 1 Model Not Found
```bash
# Train Phase 1 first
cd ../phase1_subset_20pct
python train.py

# Wait for completion, then return to Phase 2
cd ../phase2_finetune
python train.py
```

### Out of Memory
Edit `train.py` and reduce batch size:
```python
config.data.batch_size = 16  # Line 59 (instead of 32)
```

### Too Slow
Speed up training (with some accuracy loss):
```python
# Reduce input size
config.data.input_size = 128  # Line 61 (instead of 224)

# Disable RandAugment
config.data.use_randaugment = False  # Line 69

# Reduce workers
config.data.num_workers = 4  # Line 60 (instead of 8)
```

### Training Accuracy Too Low (< 50%)
Reduce augmentation strength:
```python
# Reduce Mixup
config.training.mixup_alpha = 0.1  # Line 75 (instead of 0.2)

# Reduce Cutmix
config.training.cutmix_alpha = 0.5  # Line 77 (instead of 1.0)

# Reduce RandAugment
config.data.randaugment_m = 5  # Line 71 (instead of 9)
```

## Resume Training

If training is interrupted, resume from last checkpoint:

```python
# Edit train.py, add this before training starts (around line 115)
config.training.resume_from_checkpoint = 'models/checkpoint_epoch_25.pth'
```

## Compare with Phase 1

After training completes:

```bash
# Phase 1 results
cat ../phase1_subset_20pct/logs/training_metrics.csv

# Phase 2 results
cat logs/training_metrics.csv

# Expected improvement: +10-15% top-1 accuracy
```

## Next Steps

After Phase 2 completion:

1. **Review Results**
   ```bash
   # Check final metrics
   tail -n 20 logs/imagenet_training.log
   
   # View training curves
   open logs/training_curves_*.png
   ```

2. **Test Best Model**
   ```python
   # Load and test
   checkpoint = torch.load('models/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

3. **Analyze Errors**
   - Run Grad-CAM on misclassifications
   - Identify confused classes
   - Plan Phase 3 improvements

4. **Prepare Phase 3**
   - Train on full dataset (100%)
   - Use Phase 2 best model as starting point
   - Target: 75-80% top-1 accuracy

## Configuration Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Dropout | 0.02 | Regularization |
| Mixup Alpha | 0.2 | Label smoothing |
| Cutmix Alpha | 1.0 | Spatial mixing |
| RandAugment N | 2 | # of operations |
| RandAugment M | 9 | Magnitude |
| Input Size | 224×224 | Full resolution |
| Learning Rate | 0.0001 | Fine-tuning |
| Optimizer | AdamW | Better generalization |
| Scheduler | CosineAnnealingLR | Smooth decay |
| Epochs | 50 | Sufficient convergence |
| Batch Size | 32 (eff. 128) | Memory efficient |

## Support

Having issues? Check:
1. `README.md` - Detailed documentation
2. `IMPROVEMENTS_SUMMARY.md` - Technical details
3. Phase 1 logs - Baseline comparison
4. GPU memory with `nvidia-smi`

---

**Good luck with Phase 2 training! 🚀**

