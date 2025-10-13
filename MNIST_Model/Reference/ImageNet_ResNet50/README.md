# ImageNet ResNet-50 Training Project

A modular, production-ready implementation for training ResNet-50 on ImageNet-1k dataset with phased approach optimized for cost-effective EC2 training.

## Project Overview

### Goals
- **Phase 1**: Train on 20% stratified subset → 50-60% top-1 accuracy
- **Phase 2**: Fine-tune with optimizations → 70%+ top-1 accuracy  
- **Phase 3**: Scale to full dataset → 78-80% top-1 accuracy
- **Phase 4**: Deploy best model to Hugging Face Spaces

### Key Features
- ✅ Modular OOP design with reusable components
- ✅ Gradient accumulation for limited VRAM
- ✅ Mixed precision training (AMP) for 2x speedup
- ✅ Stratified subset sampling for balanced training
- ✅ Integrated LR Finder service
- ✅ Grad-CAM analysis for model interpretability
- ✅ Automatic checkpointing with resume support
- ✅ Comprehensive logging and metrics tracking

## Project Structure

```
ImageNet_ResNet50/
├── core/                   # Core modules (reusable)
│   ├── config.py          # All configuration dataclasses
│   ├── data_manager.py    # Data loading with stratified sampling
│   ├── model.py           # ResNet-50 architecture
│   ├── trainer.py         # Training engine
│   └── logger.py          # Logging system
├── services/               # Analysis services
│   ├── lr_finder.py       # Learning rate finder
│   └── gradcam_analyzer.py # Grad-CAM analysis
├── utils/                  # Utilities
│   ├── metrics.py         # Top-K accuracy, confusion matrix
│   └── visualization.py   # Training curves, plots
├── experiments/            # Training experiments
│   ├── phase1_subset_20pct/  # Phase 1: 20% subset
│   ├── phase2_finetune/      # Phase 2: Fine-tuning
│   └── phase3_full_training/ # Phase 3: Full dataset
├── notebooks/              # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── gradcam_analysis.ipynb
└── deployment/             # Hugging Face deployment
    └── app.py             # Gradio interface
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download ImageNet Dataset

**Manual Download Required** (ImageNet-1k):
1. Register at https://image-net.org/download.php
2. Download ILSVRC2012 training and validation sets:
   - `ILSVRC2012_img_train.tar` (~138 GB)
   - `ILSVRC2012_img_val.tar` (~6.3 GB)
3. Extract to:
   ```
   data/imagenet/train/  (1000 class folders with images)
   data/imagenet/val/    (50,000 validation images)
   ```

**Validation Set Organization**:
```bash
# Use validation script to organize val images into class folders
# See: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
```

### 3. Verify Data Structure

```
data/imagenet/
├── train/
│   ├── n01440764/  (class folder)
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ... (1000 class folders)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 class folders)
```

## Usage

### Phase 1: Train on 20% Subset

```bash
cd experiments/phase1_subset_20pct
python train.py
```

**What This Does:**
- Creates stratified 20% subset (balanced across all 1000 classes)
- Trains ResNet-50 from scratch
- Uses 128x128 input size for faster iteration
- Runs for 30 epochs
- Saves best model and checkpoints
- Target: 50-60% top-1 accuracy

**Configuration:**
- Batch size: 32
- Gradient accumulation: 4 steps (effective batch = 128)
- Mixed precision: Enabled
- Optimizer: Adam
- Scheduler: OneCycleLR

**Optional: Run LR Finder First**

Uncomment lines 118-125 in `train.py` to run LR finder before training. This will automatically find optimal learning rates.

### Phase 2: Fine-Tune (Coming Next)

After Phase 1 completes:
1. Review results and training curves
2. Run Grad-CAM analysis on best model
3. Identify confused classes and augmentation improvements
4. Proceed to Phase 2 with enhanced configuration

### Phase 3: Full Dataset Training (Coming Next)

After achieving 70%+ in Phase 2:
1. Load best checkpoint from Phase 2
2. Train on full 1.28M ImageNet training set
3. Target: 78-80% top-1 accuracy

## EC2 Configuration (g4dn.xlarge)

### Instance Specs
- **GPU**: NVIDIA T4 (16 GB VRAM)
- **vCPUs**: 4
- **RAM**: 16 GB
- **Storage**: 125 GB (or add EBS volume for dataset)

### Recommended Setup

```bash
# 1. Launch g4dn.xlarge instance with Deep Learning AMI
# 2. Attach EBS volume for ImageNet data (200+ GB)
# 3. Clone repository
git clone <your-repo>
cd ImageNet_ResNet50

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download and extract ImageNet to /data/imagenet

# 6. Run Phase 1
cd experiments/phase1_subset_20pct
nohup python train.py > phase1.log 2>&1 &

# 7. Monitor progress
tail -f phase1.log
```

### Cost Optimization Tips
1. **Use Spot Instances**: 70% cost savings
2. **Phase 1 (20% subset)**: ~2-3 hours, $1-2
3. **Phase 2 (fine-tune)**: ~2-3 hours, $1-2
4. **Phase 3 (full dataset)**: ~12-15 hours, $8-12
5. **Total Estimated Cost**: $10-16 (with spot pricing)

### Checkpoint Resume

If training is interrupted:
```python
# In train.py, add:
config.training.resume_from_checkpoint = 'models/checkpoint_epoch_10.pth'
```

## Key Features Explained

### Gradient Accumulation
Simulates larger batch sizes with limited VRAM:
```python
batch_size = 32
gradient_accumulation_steps = 4
# Effective batch size = 32 * 4 = 128
```

### Mixed Precision Training
Uses automatic mixed precision (AMP) for:
- 2x faster training
- 40% memory savings
- Minimal accuracy impact

### Stratified Subset Sampling
Ensures balanced class distribution in 20% subset:
- Each class contributes exactly 20% of its samples
- Total: ~256K training images (20% of 1.28M)
- Maintains ImageNet's class distribution

### Progressive Resizing
- Phase 1: Start with 128x128 images (faster)
- Phase 2: Scale to 224x224 (standard ImageNet)
- Benefits: Faster initial training, better generalization

## Monitoring Training

### GPU Memory Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi
```

### Training Logs
```bash
# Real-time log monitoring
tail -f experiments/phase1_subset_20pct/logs/<latest>.log
```

### Training Curves
Check `logs/training_curves_*.png` for:
- Loss progression
- Top-1 and Top-5 accuracy
- Learning rate schedule
- Generalization gap

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
config.data.batch_size = 16

# Increase gradient accumulation to maintain effective batch size
config.training.gradient_accumulation_steps = 8
```

### Slow Data Loading
```python
# Increase workers
config.data.num_workers = 12

# Use persistent workers
config.data.persistent_workers = True
```

### Model Not Learning
1. Run LR Finder to find optimal learning rate
2. Check data normalization is correct
3. Verify labels are properly loaded
4. Reduce learning rate by 10x

## Performance Targets

| Phase | Dataset | Epochs | Target Top-1 | Target Top-5 | Est. Time (g4dn.xlarge) |
|-------|---------|--------|--------------|--------------|------------------------|
| 1 | 20% subset (~256K) | 30 | 50-60% | 75-85% | 2-3 hours |
| 2 | 20% subset | 30-50 | 70%+ | 88%+ | 2-3 hours |
| 3 | Full (1.28M) | 60-90 | 78-80% | 92-94% | 12-15 hours |

## References

- **ResNet Paper**: https://arxiv.org/abs/1512.03385
- **ImageNet Dataset**: https://image-net.org/
- **LR Finder**: https://arxiv.org/abs/1506.01186
- **Grad-CAM**: https://arxiv.org/abs/1610.02391

## License

This project is for educational purposes.

## Author

Krishnakanth - October 2025

---

## Quick Start Checklist

- [ ] Download ImageNet dataset
- [ ] Extract to `data/imagenet/train` and `data/imagenet/val`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run Phase 1: `cd experiments/phase1_subset_20pct && python train.py`
- [ ] Monitor GPU: `nvidia-smi`
- [ ] Review results in `logs/` directory
- [ ] Proceed to Phase 2 after validation

---

**Status**: Phase 0 & Phase 1 Implementation Complete ✅  
**Next**: Execute Phase 1 training

