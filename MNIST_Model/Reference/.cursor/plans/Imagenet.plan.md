<!-- 10c45a19-eb60-4dad-b5c7-f3d1eabaead9 2b4f12c0-25d1-4249-8891-6909ec244060 -->
# ImageNet ResNet-50 Training Project

## Overview

Create a new modular project for training ResNet-50 on ImageNet-1k dataset with phased approach: train on 20% subset first, fine-tune to 70%+ accuracy, then scale to full dataset targeting 80% top-1 accuracy. Deploy best model to Hugging Face Spaces with Gradio interface.

## Project Structure

### New Directory: `ImageNet_ResNet50/`

```
ImageNet_ResNet50/
├── core/
│   ├── __init__.py
│   ├── config.py          # All dataclass configs
│   ├── data_manager.py    # Dataset loading & augmentation
│   ├── model.py           # ResNet-50 architecture
│   ├── trainer.py         # Training logic with gradient accumulation
│   └── logger.py          # Logging system
├── services/
│   ├── __init__.py
│   ├── lr_finder.py       # Learning rate finder service
│   └── gradcam_analyzer.py # Grad-CAM analysis service
├── experiments/
│   ├── phase1_subset_20pct/
│   │   ├── train.py       # Train on 20% data
│   │   ├── logs/
│   │   └── models/
│   ├── phase2_finetune/
│   │   ├── train.py       # Fine-tune to 70%+
│   │   ├── logs/
│   │   └── models/
│   └── phase3_full_training/
│       ├── train.py       # Full dataset training
│       ├── logs/
│       └── models/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── gradcam_analysis.ipynb
├── deployment/
│   ├── app.py             # Gradio app
│   ├── model.py           # Model definition for HF
│   ├── requirements.txt
│   └── README.md
├── data/                  # ImageNet data directory
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Accuracy, loss tracking
│   └── visualization.py   # Training curves, confusion matrix
├── requirements.txt
└── README.md
```

## Implementation Plan

### Phase 0: Project Setup & Refactoring

**1. Create core modules from ERAS8 code**

Extract and refactor from `ERAS8/Model_Evolution/FineTune/`:

**a) `core/config.py`** - Consolidate all configurations:

- `DataConfig` with ImageNet normalization values
- `ModelConfig` for ResNet-50 (input_size=224x224, num_classes=1000)
- `TrainingConfig` with gradient accumulation support
- `LoggingConfig` with checkpoint management
- Add `subset_percentage` parameter for phase 1

**b) `core/data_manager.py`** - Modular data loading:

- `ImageNetDataManager` class (refactored from `CIFAR100DataManager`)
- Support for stratified sampling (20% per class)
- Albumentations augmentation pipeline for ImageNet
- Progressive resizing support (start at 128x128, scale to 224x224)
- Normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

**c) `core/model.py`** - ResNet-50 architecture:

- `ResNet50` class with BasicBlock/Bottleneck
- Support for pretrained weights (optional)
- Model from scratch option
- Parameter counting and summary methods

**d) `core/trainer.py`** - Training engine:

- `ImageNetTrainer` class with gradient accumulation (from your current implementation)
- Mixed precision training (AMP) for memory efficiency
- Automatic checkpointing every N epochs
- Resume from checkpoint support
- Early stopping logic

**e) `core/logger.py`** - Logging system:

- Extract `Logger` singleton from `cifar100_training.py`
- Add checkpoint saving/loading logs
- Track GPU memory usage

**2. Create service modules**

**a) `services/lr_finder.py`**:

- Extract LR finder logic from `Data_Exploration.ipynb`
- `LRFinder` class with range test
- Plot and save results
- Auto-suggest optimal LR

**b) `services/gradcam_analyzer.py`**:

- Extract Grad-CAM logic from `GradCAM_Analysis.ipynb`
- `GradCAMAnalyzer` class for on-demand analysis
- Support for worst/best predictions per class
- Generate HTML reports and visualizations
- Integration with training pipeline

**3. Create utility modules**

**a) `utils/metrics.py`**:

- Top-1 and Top-5 accuracy calculation
- Confusion matrix generation
- Per-class accuracy tracking

**b) `utils/visualization.py`**:

- Training curves plotting
- Confusion matrix heatmaps
- Sample prediction visualization

### Phase 1: ImageNet Data Loading & 20% Subset Training

**1. Setup ImageNet dataset**

- Create data download/preparation guide (manual download required)
- Implement stratified 20% sampling across all 1000 classes
- Validate class distribution
- Cache sampled indices for reproducibility

**2. Initial training configuration**

- Batch size: 32 (with gradient_accumulation_steps=4 → effective 128)
- Epochs: 30-50 (faster iteration on subset)
- Optimizer: Adam with OneCycleLR
- Input size: 128x128 (progressive resizing)
- Data augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, Normalize

**3. Create `experiments/phase1_subset_20pct/train.py`**

- Import core modules
- Configure for 20% subset
- Run LR finder before training
- Train ResNet-50 from scratch
- Save checkpoints every 5 epochs
- Target: ~50-60% top-1 accuracy on subset

### Phase 2: Fine-Tuning to 70%+ Accuracy

**1. Run Grad-CAM analysis on Phase 1 best model**

- Identify confused classes
- Analyze feature learning patterns
- Generate recommendations for augmentation

**2. Enhanced augmentation strategy**

- Add RandAugment or AutoAugment
- Class-specific augmentation for confused pairs
- CutMix/MixUp for better generalization
- Progressive resize to 224x224

**3. Create `experiments/phase2_finetune/train.py`**

- Load Phase 1 best checkpoint
- Apply enhanced augmentation
- Train with lower LR (fine-tuning)
- Epochs: 30-50
- Target: 70%+ top-1 accuracy

**4. Run LR finder with loaded weights**

- Find optimal LR for fine-tuning
- Adjust OneCycleLR parameters

### Phase 3: Full Dataset Training to 80%

**1. Scale to full ImageNet dataset**

- Remove subset sampling
- Increase effective batch size (gradient accumulation)
- Full 224x224 images
- Label smoothing for better calibration

**2. Create `experiments/phase3_full_training/train.py`**

- Load Phase 2 best checkpoint
- Train on full 1.28M images
- Epochs: 60-90
- Advanced techniques:
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping
  - EMA (Exponential Moving Average) of weights
  - StochasticDepth if needed

**3. Checkpoint management**

- Auto-save every 5 epochs
- Keep best 3 checkpoints
- Resume training support for EC2 interruptions

**4. Final Grad-CAM analysis**

- Analyze final model performance
- Generate comprehensive report
- Identify remaining weaknesses

### Phase 4: Deployment

**1. Create `deployment/app.py`** (Gradio):

- Similar structure to `CIFAR100HFS/app.py`
- Image upload interface
- Top-5 predictions with confidence bars
- Grad-CAM visualization toggle
- Example images from each superclass

**2. Model export**

- Clean checkpoint (weights only)
- Add model card with performance metrics
- Document architecture and training details

**3. Deploy to Hugging Face Spaces**

- Create Space repository
- Upload model (use Git LFS for large files)
- Add requirements.txt
- Test deployment

## Key Technical Decisions

### Gradient Accumulation

```python
# For g4dn.xlarge (16GB GPU)
batch_size = 32
gradient_accumulation_steps = 4
# Effective batch size = 128
```

### Mixed Precision Training

```python
# Use AMP for 2x speedup and 40% memory savings
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Checkpoint Strategy

```python
# Auto-save every 5 epochs + best model
if epoch % 5 == 0:
    save_checkpoint('epoch_{epoch}.pth')
if test_acc > best_acc:
    save_checkpoint('best_model.pth')
```

### Data Loading for EC2

```python
# Prefetch data to avoid I/O bottleneck
num_workers = 8
pin_memory = True
persistent_workers = True
```

## File References

**Refactoring sources:**

- `ERAS8/Model_Evolution/FineTune/cifar100_training.py` - trainer, logger, config
- `ERAS8/Model_Evolution/FineTune/model.py` - architecture patterns
- `ERAS8/Data_Exploration.ipynb` - LR finder cells
- `ERAS8/GradCAM_Analysis.ipynb` - Grad-CAM implementation
- `CIFAR100HFS/app.py` - deployment template

## Success Criteria

- Phase 1: 50-60% top-1 accuracy on 20% subset (validate approach)
- Phase 2: 70%+ top-1 accuracy with optimizations
- Phase 3: 78-80% top-1 accuracy on full ImageNet-1k
- Deployment: Working Gradio app on HF Spaces with Grad-CAM
- Code Quality: Fully modular, reusable OOP design

### To-dos

- [ ] Create ImageNet_ResNet50 directory structure with core/, services/, experiments/, notebooks/, deployment/, utils/ folders
- [ ] Create core/config.py with DataConfig, ModelConfig, TrainingConfig (gradient accumulation), LoggingConfig
- [ ] Create core/logger.py by extracting Logger singleton from cifar100_training.py
- [ ] Create core/data_manager.py with ImageNetDataManager supporting stratified sampling, progressive resizing, and Albumentations
- [ ] Create core/model.py with ResNet50 class (BasicBlock/Bottleneck, pretrained option, 224x224 input)
- [ ] Create core/trainer.py with ImageNetTrainer supporting gradient accumulation, mixed precision, checkpointing
- [ ] Create services/lr_finder.py by extracting LR finder from Data_Exploration.ipynb
- [ ] Create services/gradcam_analyzer.py by extracting Grad-CAM from GradCAM_Analysis.ipynb
- [ ] Create utils/metrics.py and utils/visualization.py for top-k accuracy, confusion matrix, training curves
- [ ] Create experiments/phase1_subset_20pct/train.py with stratified 20% sampling and initial training config
- [ ] Create experiments/phase2_finetune/train.py with enhanced augmentation and fine-tuning from phase1 checkpoint
- [ ] Create experiments/phase3_full_training/train.py with full dataset training, mixed precision, and checkpoint resume
- [ ] Create notebooks/data_exploration.ipynb and notebooks/gradcam_analysis.ipynb for interactive analysis
- [ ] Create deployment/app.py (Gradio) with ImageNet class names, top-5 predictions, Grad-CAM visualization
- [ ] Create requirements.txt with torch, torchvision, albumentations, gradio, pytorch-grad-cam, plotly
- [ ] Create comprehensive README.md with setup instructions, EC2 configuration, and training guide