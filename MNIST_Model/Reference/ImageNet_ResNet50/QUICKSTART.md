# Quick Start Guide - ImageNet ResNet-50 Phase 1

## Prerequisites

1. **GPU**: NVIDIA GPU with 16+ GB VRAM (e.g., T4, V100, A10)
2. **Storage**: 200+ GB for ImageNet dataset
3. **Python**: 3.8+

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd ImageNet_ResNet50
pip install -r requirements.txt
```

### 2. Download ImageNet Dataset

**Option A: Manual Download**
1. Register at https://image-net.org/download.php
2. Download files:
   - `ILSVRC2012_img_train.tar` (138 GB)
   - `ILSVRC2012_img_val.tar` (6.3 GB)
   - `ILSVRC2012_devkit_t12.tar.gz` (2.5 MB)

**Option B: Academic Torrent** (faster)
```bash
# Install academic-torrents
pip install academic-torrents

# Download (run in data/imagenet directory)
cd data/imagenet
academic-torrents download 5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5
```

### 3. Extract Dataset

```bash
cd data/imagenet

# Extract training set (takes 30-60 minutes)
mkdir train
cd train
tar -xvf ../ILSVRC2012_img_train.tar

# Extract each class tar file
for f in *.tar; do
    mkdir ${f%.tar}
    tar -xf $f -C ${f%.tar}
    rm $f
done

cd ..

# Extract validation set
mkdir val
tar -xvf ILSVRC2012_img_val.tar -C val

# Organize validation images into class folders
cd val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
```

### 4. Verify Data Structure

```bash
# Should show 1000 directories
ls data/imagenet/train | wc -l

# Should show 1000 directories
ls data/imagenet/val | wc -l
```

### 5. Run Phase 1 Training

```bash
cd experiments/phase1_subset_20pct
python train.py
```

## Configuration Options

### Enable LR Finder (Recommended for First Run)

Edit `experiments/phase1_subset_20pct/train.py`, uncomment lines 118-125:

```python
# Run LR finder before training
suggested_lr, suggested_max_lr = run_lr_finder(model, train_loader, device, config.logging.log_dir)

# Update config with suggested values
if suggested_lr:
    config.training.learning_rate = suggested_lr
    config.training.onecycle_max_lr = suggested_max_lr / 10
```

### Adjust for Your GPU

**For 8GB VRAM**:
```python
config.data.batch_size = 16
config.training.gradient_accumulation_steps = 8
config.data.input_size = 96  # Smaller images
```

**For 24GB+ VRAM**:
```python
config.data.batch_size = 64
config.training.gradient_accumulation_steps = 2
config.data.input_size = 160
```

## Monitoring Training

### Real-time Monitoring

```bash
# Terminal 1: Run training
python train.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor logs
tail -f logs/<latest-log-file>.log
```

### Check Progress

After training starts, check:
1. **GPU Utilization**: Should be 90-100%
2. **Memory Usage**: Should be ~12-14 GB (with batch_size=32)
3. **Training Speed**: ~2-3 minutes per epoch (20% subset)

## Expected Results (Phase 1)

### Baseline Performance
- **Top-1 Accuracy**: 50-60%
- **Top-5 Accuracy**: 75-85%
- **Training Time**: 2-3 hours (g4dn.xlarge)
- **Best Checkpoint**: `models/best_model.pth`

### Success Criteria
- ✅ Training completes without OOM errors
- ✅ Loss decreases smoothly
- ✅ Top-1 accuracy > 50%
- ✅ Top-5 accuracy > 75%
- ✅ No excessive overfitting (gap < 15%)

## Outputs

After training, check these files:

```
experiments/phase1_subset_20pct/
├── logs/
│   ├── 20251013_HHMMSS_imagenet_training.log  # Detailed log
│   ├── training_curves_*.png                   # Visual progress
│   ├── lr_finder_results.png                   # LR finder plot
│   └── training_metrics.csv                    # Epoch-by-epoch data
└── models/
    ├── best_model.pth                          # Best performing model
    ├── checkpoint_epoch_5.pth                  # Periodic checkpoints
    ├── checkpoint_epoch_10.pth
    └── checkpoint_epoch_15.pth
```

## Next Steps After Phase 1

1. **Review Results**
   ```bash
   # Check training curves
   open logs/training_curves_*.png
   
   # Read final metrics
   tail -100 logs/*.log
   ```

2. **Run Grad-CAM Analysis**
   ```bash
   # Coming in Phase 2
   # Use notebooks/gradcam_analysis.ipynb
   ```

3. **Proceed to Phase 2**
   - Update augmentation based on Grad-CAM insights
   - Fine-tune with 224x224 images
   - Target 70%+ accuracy

## Troubleshooting

### Issue: FileNotFoundError (ImageNet not found)
**Solution**: Verify data structure matches expected layout

### Issue: CUDA Out of Memory
**Solutions**:
```python
# Reduce batch size
config.data.batch_size = 16

# Or reduce input size
config.data.input_size = 96
```

### Issue: Slow Training
**Solutions**:
- Increase `num_workers` to 12-16
- Use SSD/NVMe for dataset storage
- Enable `persistent_workers=True`

### Issue: Poor Accuracy
**Solutions**:
- Run LR Finder to optimize learning rate
- Check data augmentation isn't too aggressive
- Verify normalization values are correct

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review training curves
3. Verify GPU memory usage with `nvidia-smi`

---

**Ready to Train!** 🚀

Run `python experiments/phase1_subset_20pct/train.py` to begin.

