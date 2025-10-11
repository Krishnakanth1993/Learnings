# CIFAR-100 Training Experiments - Fine-Tuning Phase
*This README documents the fine-tuning experiments for CIFAR-100 classification using ResNet-18 with Bottleneck blocks and advanced regularization.*

## Project Summary

### Target
- ✅ Implement ResNet-18 with Bottleneck blocks
- ✅ Add comprehensive dropout regularization
- ✅ Use Adam optimizer with ReduceLROnPlateau scheduler
- ✅ Apply Albumentations data augmentation
- ✅ Reduce model parameters (~1M vs 23M)
- ✅ Improve generalization and reduce overfitting

### Result
- **Parameters**: 929,572 (~0.93M) - **96% reduction from Optimize phase**
- **Best Test Accuracy**: 69.28% (epoch 99)
- **Final Test Accuracy**: 69.24% (epoch 100)
- **Final Train Accuracy**: 90.71%
- **Final Train-Test Gap**: 21.47% (improved from 25.65%)
- **Epochs Completed**: 100
- **Overfitting Epochs**: 26 (vs 81 in Optimize phase)

### Key Improvements Over Optimize Phase
✅ **96% parameter reduction**   
✅ **Dropout regularization added**   
✅ **Better architecture efficiency**   

## Resources

### Files
- **Training Log**: [20251009_104659_cifar100_training.log](Model_Evolution/FineTune/logs/20251009_104659_cifar100_training.log)
- **Training Curves**: [training_curves_20251009_122012.png](Model_Evolution/FineTune/logs/training_curves_20251009_122012.png)
- **Saved Model**: [cifar100_model_20251009_194111.pth](Model_Evolution/FineTune/models/cifar100_model_20251009_194111.pth)
- **Model Code**: [model.py](Model_Evolution/FineTune/model.py)
- **Training Script**: [cifar100_training.py](Model_Evolution/FineTune/cifar100_training.py)

### Key Metrics Summary

**Training Duration**: ~1.5 hours (100 epochs)  
**Hardware**: CUDA GPU  
**Framework**: PyTorch + Albumentations

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 69.28% @ epoch 99 |
| **Final Test Accuracy** | 69.24% @ epoch 100 |
| **Final Train Accuracy** | 90.71% @ epoch 100 |
| **Final Gap** | 21.47% |
| **Average Gap** | 9.13% |
| **Overfitting Epochs** | 26 |
| **Parameters** | 929,572 |
| **Model Size** | 3.55 MB |
| **Params/Sample Ratio** | 18.6:1 |

### Visual Analysis (from training curves)

**Expected Curves**:
- **Train Loss**: Smooth decrease to ~0.31
- **Test Loss**: Decrease to ~1.22, plateau after epoch 60
- **Train Acc**: Smooth increase to 90.71%
- **Test Acc**: Increase to 69.28%, plateau after epoch 60
- **Gap**: Widens after epoch 75 (LR reductions)

================================================================================
### GRAD-CAM ANALYSIS
================================================================================

📁 OUTPUT FILES GENERATED:
  1. **Confusion Matrix**: [link here](/gradcam_results/confusion_matrix_full.png)
  2. **Per-Class Accuracy CSV**: [link here](/gradcam_results/per_class_accuracy.csv)
  3. **Interactive Accuracy Chart**: [link here](/gradcam_results/per_class_accuracy_interactive.html)
  4. **Confused Pairs CSV**: [link here](/gradcam_results/most_confused_pairs.csv)
  5. **Confused Pairs Chart**: [link here](/gradcam_results/top_confused_pairs.png)
  6. **Per-Class Grad-CAM Images**: [link here](/gradcam_results/worst_predictions/)
  7. **Confused Pair Examples**: [link here](/gradcam_results/confused_pair_*.png)
  8. **Analysis Summary**: [link here](/gradcam_results/analysis_summary.txt)

🎯 KEY FINDINGS:
  - Test Accuracy: 73.56%
  - Main Issue: Within-category confusions (7/20)
  - High-confidence errors: 806 cases
  - Worst performing category: Check visualization above

💡 USAGE:
  1. Use interactive widget above to explore any class
  2. Review saved PNG files for offline analysis
  3. Open HTML file in browser for interactive charts
  4. Read recommendations.txt for next steps

✅ Analysis pipeline completed successfully!

Input: 32×32×3 RGB images
├── Initial Conv Block
│ ├── Conv2d: 3→64, 3×3, stride=1, padding=1
│ ├── BatchNorm2d(64)
│ └── ReLU
│
├── Layer 1: [64, 64, 128] - 32×32 (2 blocks)
│ ├── BottleneckBlock-1: 64→16→16→64 (expansion: 4x)
│ │ ├── Conv2d: 64→16, 1×1 (reduce)
│ │ ├── Conv2d: 16→16, 3×3 (process)
│ │ └── Conv2d: 16→64, 1×1 (expand)
│ ├── BottleneckBlock-2: 64→16→16→64
│ └── Dropout(0.05)
│
├── Layer 2: [128, 32, 32, 128] - 16×16 (2 blocks)
│ ├── BottleneckBlock-1: 64→32→32→128, stride=2 (downsample)
│ │ └── Skip: Conv2d 64→128, 1×1, stride=2
│ ├── BottleneckBlock-2: 128→32→32→128
│ └── Dropout(0.05)
│
├── Layer 3: [256, 64, 64, 256] - 8×8 (2 blocks)
│ ├── BottleneckBlock-1: 128→64→64→256, stride=2 (downsample)
│ │ └── Skip: Conv2d 128→256, 1×1, stride=2
│ ├── BottleneckBlock-2: 256→64→64→256
│ └── Dropout(0.05)
│
├── Layer 4: [512, 128, 128, 512] - 4×4 (2 blocks)
│ ├── BottleneckBlock-1: 256→128→128→512, stride=2 (downsample)
│ │ └── Skip: Conv2d 256→512, 1×1, stride=2
│ ├── BottleneckBlock-2: 512→128→128→512
│ └── Dropout(0.05)
│
├── Classifier
│ ├── AdaptiveAvgPool2d(1, 1)
│ ├── Dropout(0.05)
│ └── Linear: 512 → 100
│
Output: 100 class probabilities (log_softmax)
This README provides a comprehensive analysis of your FineTune experiment, showing both the massive parameter reduction achieved and the clear path forward to improve test accuracy through better regularization!


**Bottleneck Block Design**:
```python
# Bottleneck: expand → process → reduce
in_channels → in_channels/4 → in_channels/4 → out_channels
    (1×1)          (3×3)              (1×1)
```

**Key Features**:
- **Bottleneck blocks** for parameter efficiency
- **4 dropout layers** after each residual layer (0.05 rate)
- **1 FC dropout** before final classifier (0.05 rate)
- **Progressive channels**: 64 → 128 → 256 → 512
- **Expansion ratio**: 4× in bottleneck blocks
- **Residual connections** with proper downsampling
- **Batch Normalization** after every convolution
- **Kaiming initialization** for weights

**Architecture Comparison**:
| Metric | Optimize (Exp-1) | FineTune (FT-1) | Improvement |
|--------|------------------|-----------------|-------------|
| Parameters | 23.18M | 0.93M | **-96.0%** |
| Layers | 16 residual blocks | 8 bottleneck blocks | Simpler |
| Dropout | ❌ None | ✅ 5 layers | Added |
| Model Size | 88.43 MB | 3.55 MB | **-96.0%** |
| Architecture | Custom ResNet | ResNet-18 Bottleneck | Standard |

#### Training Configuration

- **Epochs**: 100 (completed full training)
- **Optimizer**: Adam
  - Learning Rate: 0.00251 (found via LR Finder)
  - Betas: (0.9, 0.999)
  - Eps: 1e-08
  - Weight Decay: 0.0001
- **Scheduler**: ReduceLROnPlateau
  - Mode: min (based on test loss)
  - Factor: 0.5
  - Patience: 10 epochs
  - Threshold: 0.0001
- **LR Schedule**: 
  - 0.00251 (epochs 1-61)
  - 0.001255 (epochs 62-73) - 1st reduction
  - 0.000628 (epochs 74-86) - 2nd reduction
  - 0.000314 (epochs 87-97) - 3rd reduction
  - 0.000157 (epochs 98-100) - 4th reduction
- **Batch Size**: 128
- **Dropout Rate**: 0.05 (conv layers + FC layer)
- **Early Stopping**: Stop if train_acc - test_acc > 15% for 10+ epochs

#### Data Configuration

**Dataset**: CIFAR-100  
**Train Samples**: 50,000  
**Test Samples**: 10,000  
**Image Size**: 32×32 RGB  

**Normalization**:
- Mean: (0.5071, 0.4867, 0.4409)
- Std: (0.2673, 0.2564, 0.2762)

**Data Augmentation** (Albumentations):
- HorizontalFlip (p=0.5)
- ShiftScaleRotate (shift=0.1, scale=0.1, rotate=±7°)
- CoarseDropout (1 hole, 16×16)
- Normalize (CIFAR-100 mean/std)

#### Results

**Performance Metrics**:
- **Best Test Accuracy**: 69.28% (epoch 99)
- **Final Test Accuracy**: 69.24% (epoch 100)
- **Final Train Accuracy**: 90.71%
- **Best Train Accuracy**: 90.71% (epoch 100)
- **Final Train Loss**: 0.3142
- **Final Test Loss**: 1.2220

**Top-1 Metrics**:
- **Top-1 Accuracy**: 69.28% (best)
- **Top-1 Error**: 30.72%

**Generalization Analysis**:
- **Final Train-Test Gap**: 21.47% (train > test) - **Overfitting**
- **Maximum Gap**: 21.47% (epoch 100)
- **Average Gap**: 9.13% across all epochs
- **Overfitting Started**: After epoch 75 (gap exceeded 15%)
- **Consecutive Overfitting Epochs**: 26 out of 100

**Training Progression**:
| Epoch Range | Train Acc | Test Acc | Gap | LR | Phase |
|-------------|-----------|----------|-----|-----|-------|
| 1-10 | 9.34% → 48.02% | 14.32% → 47.15% | -4.98% → 0.87% | 0.00251 | **Healthy learning** |
| 11-20 | 49.74% → 59.05% | 50.43% → 55.30% | -0.69% → 3.75% | 0.00251 | **Good progress** |
| 21-30 | 59.35% → 63.85% | 55.29% → 58.28% | 4.06% → 5.57% | 0.00251 | **Slight overfitting** |
| 31-40 | 64.27% → 66.56% | 59.21% → 60.71% | 5.06% → 5.85% | 0.00251 | **Manageable gap** |
| 41-50 | 66.83% → 68.66% | 59.95% → 59.44% | 6.88% → 9.22% | 0.00251 | **Gap increasing** |
| 51-61 | 68.55% → 70.46% | 62.26% → 61.85% | 6.29% → 8.61% | 0.00251 | **Pre-reduction** |
| 62-73 | 70.15% → 78.87% | 62.22% → 66.27% | 7.93% → 12.60% | 0.001255 | **1st LR drop** |
| 74-86 | 78.99% → 85.36% | 66.51% → 67.86% | 12.48% → 17.50% | 0.000628 | **2nd LR drop** |
| 87-97 | 85.64% → 89.30% | 67.61% → 68.80% | 18.03% → 20.50% | 0.000314 | **3rd LR drop** |
| 98-100 | 89.44% → 90.71% | 68.72% → 69.24% | 20.72% → 21.47% | 0.000157 | **4th LR drop** |

**Learning Rate Reductions**:
1. **Epoch 62**: 0.00251 → 0.001255 (test loss plateaued at ~1.36)
2. **Epoch 74**: 0.001255 → 0.000628 (test loss plateaued at ~1.25)
3. **Epoch 87**: 0.000628 → 0.000314 (test loss plateaued at ~1.24)
4. **Epoch 98**: 0.000314 → 0.000157 (test loss plateaued at ~1.25)

#### Key Observations

**✅ Successes**:
1. **Massive Parameter Reduction**: 96% fewer parameters (23.18M → 0.93M)
2. **Efficient Architecture**: Bottleneck blocks achieve good performance with fewer params
3. **Dropout Works**: Added regularization reduced overfitting epochs (81 → 26)
4. **Fast Initial Learning**: 47.15% test accuracy in just 10 epochs
5. **Smooth Training**: No catastrophic failures or divergence
6. **Better Generalization Early**: First 60 epochs show healthy train-test balance
7. **Proper LR Finding**: Starting LR 0.00251 from LR Finder experiment
8. **Complete Training**: Ran full 100 epochs without crashes

**⚠️ Areas of Concern**:
1. **Still Overfitting**: 21.47% gap is high (though better than 25.65%)
2. **Test Accuracy Lower**: 69.24% vs 74.61% from Optimize phase
3. **Overfitting After Epoch 75**: Last 25 epochs show severe overfitting
4. **LR Reductions Help Train More Than Test**: Each reduction → train jumps, test small gain
5. **Low Dropout Rate**: 0.05 may be too conservative
6. **Plateau Pattern**: Test loss stuck around 1.20-1.25 after epoch 60

**Pattern Analysis**:
- **Epochs 1-61**: Excellent learning phase (gap < 10%)
- **Epochs 62-74**: First LR reduction → train jumps, gap widens (7.93% → 12.48%)
- **Epochs 75-100**: Overfitting phase begins (gap 15.11% → 21.47%)
- **Effect of LR Reductions**: Helps train accuracy more than test accuracy

---

## Comparative Analysis: Optimize vs FineTune

### Performance Comparison

| Metric | Optimize (Exp-1) | FineTune (FT-1) | Δ Change |
|--------|------------------|------------------|----------|
| **Parameters** | 23,182,440 | 929,572 | **-96.0%** ✅ |
| **Model Size** | 88.43 MB | 3.55 MB | **-96.0%** ✅ |
| **Best Test Acc** | 74.61% | 69.28% | -5.33% ⚠️ |
| **Final Test Acc** | 73.97% | 69.24% | -4.73% ⚠️ |
| **Final Train Acc** | 99.62% | 90.71% | **-8.91%** ✅ |
| **Final Gap** | 25.65% | 21.47% | **-4.18%** ✅ |
| **Avg Gap** | 20.61% | 9.13% | **-11.48%** ✅ |
| **Overfitting Epochs** | 81 | 26 | **-68%** ✅ |
| **Initial LR** | 0.001 | 0.00251 | +151% |
| **Dropout** | ❌ None | ✅ 5 layers | Added ✅ |

### Architecture Comparison

**Optimize (Exp-1) - Custom ResNet**:
- 16 residual blocks
- No bottleneck compression
- Channels: 64→128→256→512
- Final layer: 512→1000 (oversized)
- No dropout regularization
- 23.18M parameters

**FineTune (FT-1) - ResNet-18 Bottleneck**:
- 8 bottleneck blocks (4 layers × 2 blocks)
- 1×1 conv compression (4× expansion)
- Channels: 64→128→256→512
- Final layer: 512→100 (right-sized)
- 5 dropout layers (0.05 rate)
- 0.93M parameters

### Trade-off Analysis

**What We Gained** ✅:
1. **96% fewer parameters** → Much faster training & inference
2. **96% smaller model** → Easier deployment
3. **68% less overfitting** → Better generalization potential
4. **11.48% lower average gap** → More stable training
5. **Standard architecture** → Easier to understand & maintain
6. **Dropout regularization** → Foundation for improvement

**What We Lost** ⚠️:
1. **5.33% test accuracy** → Need to recover this
2. **Model capacity** → May be too small for CIFAR-100

**Net Assessment**: 
- **Trade-off is ACCEPTABLE** - Massive efficiency gains for modest accuracy loss
- **Path forward is clear** - Increase dropout, try other regularization
- **Architecture is better** - Standard, efficient, maintainable

---

## Detailed Training Analysis

### Phase 1: Healthy Learning (Epochs 1-61)

**Characteristics**:
- Train: 9.34% → 70.46%
- Test: 14.32% → 61.85%
- Gap: -4.98% → 8.61%
- Status: ✅ **Excellent learning phase**

**Key Milestones**:
- **Epoch 10**: 48.02% test acc (fast start)
- **Epoch 20**: 55.30% test acc
- **Epoch 30**: 58.28% test acc
- **Epoch 40**: 60.71% test acc
- **Epoch 50**: 59.44% test acc (slight dip)
- **Epoch 60**: 62.97% test acc (best so far)

**Analysis**: Model learning generalizable features effectively. Gap stays under 10% for first 60 epochs.

### Phase 2: First LR Reduction (Epochs 62-73)

**Trigger**: Test loss plateaued around 1.36 for 10 epochs  
**LR Change**: 0.00251 → 0.001255 (50% reduction)

**Impact**:
- Train jumped: 70.15% → 78.87% (+8.72%)
- Test improved: 62.22% → 66.27% (+4.05%)
- Gap widened: 7.93% → 12.60% (+4.67%)

**Analysis**: LR reduction helped but train benefited more than test. Gap increasing.

### Phase 3: Second LR Reduction (Epochs 74-86)

**Trigger**: Test loss plateaued around 1.25 for 10 epochs  
**LR Change**: 0.001255 → 0.000628 (50% reduction)

**Impact**:
- Train jumped: 78.99% → 85.36% (+6.37%)
- Test improved: 66.51% → 67.86% (+1.35%)
- Gap widened: 12.48% → 17.50% (+5.02%)

**Overfitting Begins**: Epoch 75 gap exceeded 15% threshold

**Analysis**: Severe imbalance - train improving 4.7× faster than test. Clear overfitting.

### Phase 4: Third LR Reduction (Epochs 87-97)

**Trigger**: Test loss plateaued around 1.24 for 10 epochs  
**LR Change**: 0.000628 → 0.000314 (50% reduction)

**Impact**:
- Train jumped: 85.64% → 89.30% (+3.66%)
- Test improved: 67.61% → 68.80% (+1.19%)
- Gap widened: 18.03% → 20.50% (+2.47%)

**Analysis**: Pattern continues - train improving 3× faster than test. Overfitting worsening.

### Phase 5: Fourth LR Reduction (Epochs 98-100)

**Trigger**: Test loss plateaued around 1.25 for 10 epochs  
**LR Change**: 0.000314 → 0.000157 (50% reduction)

**Impact**:
- Train: 89.44% → 90.71% (+1.27%)
- Test: 68.72% → 69.24% (+0.52%)
- Gap: 20.72% → 21.47% (+0.75%)

**Analysis**: Both improving slowly. Too late to fix overfitting.

### Learning Rate Effectiveness Analysis

| LR | Epoch Range | Train Δ | Test Δ | Train/Test Ratio | Effectiveness |
|----|-------------|---------|--------|------------------|---------------|
| 0.00251 | 1-61 | +61.12% | +47.53% | 1.29× | ✅ Excellent |
| 0.001255 | 62-73 | +8.72% | +4.05% | 2.15× | ⚠️ Imbalanced |
| 0.000628 | 74-86 | +6.37% | +1.35% | 4.72× | ❌ Overfitting |
| 0.000314 | 87-97 | +3.66% | +1.19% | 3.08× | ❌ Overfitting |
| 0.000157 | 98-100 | +1.27% | +0.52% | 2.44× | ❌ Minimal |

**Conclusion**: Initial LR (0.00251) was perfect. Reductions helped train more than test, worsening overfitting.

---
---



Detailed Log:

2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO - Logger initialized. Log file: C:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS8\Model_Evolution\FineTune\logs\20251009_104659_cifar100_training.log
2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO - Updated Configuration (from main()):
2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO -   - Epochs: 100
2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO -   - Learning Rate: 0.00251
2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO -   - Optimizer: Adam
2025-10-09 10:46:59,161 - CIFAR-100_Training - INFO -   - Weight Decay: 0.0001
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Adam Betas: (0.9, 0.999)
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Adam Eps: 1e-08
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Scheduler: ReduceLROnPlateau
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Mode: min
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Factor: 0.5
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Patience: 10
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Threshold: 0.0001
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Batch Size: 128
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Num Workers: 4
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Pin Memory: True
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Shuffle: True
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Dropout Rate: 0.05
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Device: CUDA
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Log Directory: C:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS8\Model_Evolution\FineTune\logs
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Model Save Directory: C:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS8\Model_Evolution\FineTune\models
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Save Model: True
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO -   - Log Level: DEBUG
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - CIFAR-100 TRAINING EXPERIMENT STARTED
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - Data Config: DataConfig(data_dir='./data', batch_size=128, num_workers=4, pin_memory=True, shuffle=True, cifar100_mean=(0.507076, 0.48655, 0.440919), cifar100_std=(0.267334, 0.256438, 0.27615), rotation_range=(-7.0, 7.0), fill_value=1)
2025-10-09 10:46:59,163 - CIFAR-100_Training - INFO - Model Config: ModelConfig(input_channels=3, input_size=(32, 32), num_classes=100, dropout_rate=0.05)
2025-10-09 10:46:59,165 - CIFAR-100_Training - INFO - Training Config: TrainingConfig(epochs=100, learning_rate=0.00251, momentum=0.9, weight_decay=0.0001, scheduler_step_size=10, scheduler_gamma=0.1, seed=1, optimizer_type='Adam', adam_betas=(0.9, 0.999), adam_eps=1e-08, rmsprop_alpha=0.99, scheduler_type='ReduceLROnPlateau', cosine_t_max=20, exponential_gamma=0.95, plateau_mode='min', plateau_factor=0.5, plateau_patience=10, plateau_threshold=0.0001)
2025-10-09 10:46:59,165 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:46:59,165 - CIFAR-100_Training - INFO - Setting up data...
2025-10-09 10:46:59,165 - CIFAR-100_Training - INFO - Using Albumentations for data augmentation
2025-10-09 10:46:59,167 - CIFAR-100_Training - INFO - Loading CIFAR-100 dataset...
2025-10-09 10:47:00,472 - CIFAR-100_Training - INFO - CIFAR-100 dataset loaded successfully!
2025-10-09 10:47:00,472 - CIFAR-100_Training - INFO - Train samples: 50000
2025-10-09 10:47:00,472 - CIFAR-100_Training - INFO - Test samples: 10000
2025-10-09 10:47:00,472 - CIFAR-100_Training - INFO - Augmentation library: Albumentations
2025-10-09 10:47:00,472 - CIFAR-100_Training - INFO - Computing CIFAR-100 data statistics...
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO - CIFAR-100 Data Statistics:
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Shape: (50000, 32, 32, 3)
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Size: 153,600,000
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Min: 0.0000
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Max: 1.0000
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Mean: 0.4782
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Std: 0.2682
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO -   - Variance: 0.0719
2025-10-09 10:47:02,291 - CIFAR-100_Training - INFO - Channel-wise Statistics:
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -   Red Channel:
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Mean: 0.5071
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Std: 0.2673
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Min: 0.0000
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Max: 1.0000
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -   Green Channel:
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Mean: 0.4865
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Std: 0.2564
2025-10-09 10:47:02,301 - CIFAR-100_Training - INFO -     - Min: 0.0000
2025-10-09 10:47:02,306 - CIFAR-100_Training - INFO -     - Max: 1.0000
2025-10-09 10:47:02,306 - CIFAR-100_Training - INFO -   Blue Channel:
2025-10-09 10:47:02,306 - CIFAR-100_Training - INFO -     - Mean: 0.4409
2025-10-09 10:47:02,306 - CIFAR-100_Training - INFO -     - Std: 0.2762
2025-10-09 10:47:02,307 - CIFAR-100_Training - INFO -     - Min: 0.0000
2025-10-09 10:47:02,307 - CIFAR-100_Training - INFO -     - Max: 1.0000
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO - CIFAR-100 Batch Information:
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO -   - Batch size: 128
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO -   - Image shape: torch.Size([3, 32, 32])
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO -   - Label shape: torch.Size([128])
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO -   - Data type: torch.float32
2025-10-09 10:47:14,708 - CIFAR-100_Training - INFO -   - Number of classes: 100
2025-10-09 10:47:15,755 - CIFAR-100_Training - INFO - Getting input size from CIFAR-100 data loader...
2025-10-09 10:47:27,530 - CIFAR-100_Training - INFO - CIFAR-100 input size from data loader: (3, 32, 32)
2025-10-09 10:47:28,343 - CIFAR-100_Training - INFO - Setting up model...
2025-10-09 10:47:28,379 - CIFAR-100_Training - INFO - Generating ResNet-18 with Bottleneck summary...
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - ResNet-18 with Bottleneck Architecture Summary:
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   - Total Parameters: 929,572
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   - Batch Normalization: Yes
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   - Dropout: Yes
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   - FC Layers: Yes
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   - GAP Layers: Yes
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - DETAILED MODEL ARCHITECTURE SUMMARY
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -         Layer (type)               Output Shape         Param #
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO - ================================================================
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -             Conv2d-1           [-1, 64, 32, 32]           1,728
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -        BatchNorm2d-2           [-1, 64, 32, 32]             128
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -               ReLU-3           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -             Conv2d-4           [-1, 16, 32, 32]           1,024
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -        BatchNorm2d-5           [-1, 16, 32, 32]              32
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -               ReLU-6           [-1, 16, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -             Conv2d-7           [-1, 16, 32, 32]           2,304
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -        BatchNorm2d-8           [-1, 16, 32, 32]              32
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -               ReLU-9           [-1, 16, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-10           [-1, 64, 32, 32]           1,024
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-11           [-1, 64, 32, 32]             128
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-12           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   BottleneckBlock-13           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-14           [-1, 16, 32, 32]           1,024
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-15           [-1, 16, 32, 32]              32
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-16           [-1, 16, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-17           [-1, 16, 32, 32]           2,304
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-18           [-1, 16, 32, 32]              32
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-19           [-1, 16, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-20           [-1, 64, 32, 32]           1,024
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-21           [-1, 64, 32, 32]             128
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-22           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   BottleneckBlock-23           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -           Dropout-24           [-1, 64, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-25           [-1, 32, 32, 32]           2,048
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-26           [-1, 32, 32, 32]              64
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-27           [-1, 32, 32, 32]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-28           [-1, 32, 16, 16]           9,216
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-29           [-1, 32, 16, 16]              64
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-30           [-1, 32, 16, 16]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-31          [-1, 128, 16, 16]           4,096
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-32          [-1, 128, 16, 16]             256
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-33          [-1, 128, 16, 16]           8,192
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -       BatchNorm2d-34          [-1, 128, 16, 16]             256
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -              ReLU-35          [-1, 128, 16, 16]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -   BottleneckBlock-36          [-1, 128, 16, 16]               0
2025-10-09 10:47:28,872 - CIFAR-100_Training - INFO -            Conv2d-37           [-1, 32, 16, 16]           4,096
2025-10-09 10:47:28,882 - CIFAR-100_Training - INFO -       BatchNorm2d-38           [-1, 32, 16, 16]              64
2025-10-09 10:47:28,882 - CIFAR-100_Training - INFO -              ReLU-39           [-1, 32, 16, 16]               0
2025-10-09 10:47:28,882 - CIFAR-100_Training - INFO -            Conv2d-40           [-1, 32, 16, 16]           9,216
2025-10-09 10:47:28,882 - CIFAR-100_Training - INFO -       BatchNorm2d-41           [-1, 32, 16, 16]              64
2025-10-09 10:47:28,882 - CIFAR-100_Training - INFO -              ReLU-42           [-1, 32, 16, 16]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -            Conv2d-43          [-1, 128, 16, 16]           4,096
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -       BatchNorm2d-44          [-1, 128, 16, 16]             256
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -              ReLU-45          [-1, 128, 16, 16]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -   BottleneckBlock-46          [-1, 128, 16, 16]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -           Dropout-47          [-1, 128, 16, 16]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -            Conv2d-48           [-1, 64, 16, 16]           8,192
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -       BatchNorm2d-49           [-1, 64, 16, 16]             128
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -              ReLU-50           [-1, 64, 16, 16]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -            Conv2d-51             [-1, 64, 8, 8]          36,864
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -       BatchNorm2d-52             [-1, 64, 8, 8]             128
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -              ReLU-53             [-1, 64, 8, 8]               0
2025-10-09 10:47:28,883 - CIFAR-100_Training - INFO -            Conv2d-54            [-1, 256, 8, 8]          16,384
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -       BatchNorm2d-55            [-1, 256, 8, 8]             512
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -            Conv2d-56            [-1, 256, 8, 8]          32,768
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -       BatchNorm2d-57            [-1, 256, 8, 8]             512
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -              ReLU-58            [-1, 256, 8, 8]               0
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -   BottleneckBlock-59            [-1, 256, 8, 8]               0
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -            Conv2d-60             [-1, 64, 8, 8]          16,384
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -       BatchNorm2d-61             [-1, 64, 8, 8]             128
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -              ReLU-62             [-1, 64, 8, 8]               0
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -            Conv2d-63             [-1, 64, 8, 8]          36,864
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -       BatchNorm2d-64             [-1, 64, 8, 8]             128
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -              ReLU-65             [-1, 64, 8, 8]               0
2025-10-09 10:47:28,885 - CIFAR-100_Training - INFO -            Conv2d-66            [-1, 256, 8, 8]          16,384
2025-10-09 10:47:28,886 - CIFAR-100_Training - INFO -       BatchNorm2d-67            [-1, 256, 8, 8]             512
2025-10-09 10:47:28,886 - CIFAR-100_Training - INFO -              ReLU-68            [-1, 256, 8, 8]               0
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -   BottleneckBlock-69            [-1, 256, 8, 8]               0
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -           Dropout-70            [-1, 256, 8, 8]               0
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -            Conv2d-71            [-1, 128, 8, 8]          32,768
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -       BatchNorm2d-72            [-1, 128, 8, 8]             256
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -              ReLU-73            [-1, 128, 8, 8]               0
2025-10-09 10:47:28,887 - CIFAR-100_Training - INFO -            Conv2d-74            [-1, 128, 4, 4]         147,456
2025-10-09 10:47:28,888 - CIFAR-100_Training - INFO -       BatchNorm2d-75            [-1, 128, 4, 4]             256
2025-10-09 10:47:28,888 - CIFAR-100_Training - INFO -              ReLU-76            [-1, 128, 4, 4]               0
2025-10-09 10:47:28,888 - CIFAR-100_Training - INFO -            Conv2d-77            [-1, 512, 4, 4]          65,536
2025-10-09 10:47:28,888 - CIFAR-100_Training - INFO -       BatchNorm2d-78            [-1, 512, 4, 4]           1,024
2025-10-09 10:47:28,888 - CIFAR-100_Training - INFO -            Conv2d-79            [-1, 512, 4, 4]         131,072
2025-10-09 10:47:28,890 - CIFAR-100_Training - INFO -       BatchNorm2d-80            [-1, 512, 4, 4]           1,024
2025-10-09 10:47:28,890 - CIFAR-100_Training - INFO -              ReLU-81            [-1, 512, 4, 4]               0
2025-10-09 10:47:28,890 - CIFAR-100_Training - INFO -   BottleneckBlock-82            [-1, 512, 4, 4]               0
2025-10-09 10:47:28,890 - CIFAR-100_Training - INFO -            Conv2d-83            [-1, 128, 4, 4]          65,536
2025-10-09 10:47:28,895 - CIFAR-100_Training - INFO -       BatchNorm2d-84            [-1, 128, 4, 4]             256
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -              ReLU-85            [-1, 128, 4, 4]               0
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -            Conv2d-86            [-1, 128, 4, 4]         147,456
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -       BatchNorm2d-87            [-1, 128, 4, 4]             256
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -              ReLU-88            [-1, 128, 4, 4]               0
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -            Conv2d-89            [-1, 512, 4, 4]          65,536
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -       BatchNorm2d-90            [-1, 512, 4, 4]           1,024
2025-10-09 10:47:28,896 - CIFAR-100_Training - INFO -              ReLU-91            [-1, 512, 4, 4]               0
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO -   BottleneckBlock-92            [-1, 512, 4, 4]               0
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO -           Dropout-93            [-1, 512, 4, 4]               0
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO - AdaptiveAvgPool2d-94            [-1, 512, 1, 1]               0
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO -           Dropout-95                  [-1, 512]               0
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO -            Linear-96                  [-1, 100]          51,300
2025-10-09 10:47:28,897 - CIFAR-100_Training - INFO - ================================================================
2025-10-09 10:47:28,898 - CIFAR-100_Training - INFO - Total params: 929,572
2025-10-09 10:47:28,898 - CIFAR-100_Training - INFO - Trainable params: 929,572
2025-10-09 10:47:28,898 - CIFAR-100_Training - INFO - Non-trainable params: 0
2025-10-09 10:47:28,898 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
2025-10-09 10:47:28,899 - CIFAR-100_Training - INFO - Input size (MB): 0.01
2025-10-09 10:47:28,899 - CIFAR-100_Training - INFO - Forward/backward pass size (MB): 14.62
2025-10-09 10:47:28,899 - CIFAR-100_Training - INFO - Params size (MB): 3.55
2025-10-09 10:47:28,899 - CIFAR-100_Training - INFO - Estimated Total Size (MB): 18.18
2025-10-09 10:47:28,899 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
2025-10-09 10:47:28,901 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 10:47:28,901 - CIFAR-100_Training - INFO - Setting up trainer...
2025-10-09 10:47:28,903 - CIFAR-100_Training - INFO - Using device: cuda
2025-10-09 10:47:28,903 - CIFAR-100_Training - INFO - Early stopping: Stop if train_acc - test_acc > 15.0% for 10 epochs
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO - Starting training process...
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO - Starting training process...
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO - Using optimizer: Adam
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO - Using scheduler: ReduceLROnPlateau
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO - Optimizer Configuration:
2025-10-09 10:47:28,905 - CIFAR-100_Training - INFO -   - Learning Rate: 0.00251
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Betas: (0.9, 0.999)
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Eps: 1e-08
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Weight Decay: 0.0001
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO - Scheduler Configuration:
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Mode: min
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Factor: 0.5
2025-10-09 10:47:28,907 - CIFAR-100_Training - INFO -   - Patience: 10
2025-10-09 10:47:28,909 - CIFAR-100_Training - INFO -   - Threshold: 0.0001
2025-10-09 10:47:28,909 - CIFAR-100_Training - INFO - Starting Epoch 1/100
2025-10-09 10:48:24,781 - CIFAR-100_Training - INFO - Epoch  1: Train Loss: 3.9206, Train Acc: 9.34%, Test Loss: 3.6450, Test Acc: 14.32%, Acc Diff: -4.98%, LR: 0.002510
2025-10-09 10:48:24,781 - CIFAR-100_Training - INFO - Starting Epoch 2/100
2025-10-09 10:49:20,295 - CIFAR-100_Training - INFO - Epoch  2: Train Loss: 3.3155, Train Acc: 18.81%, Test Loss: 3.1224, Test Acc: 22.38%, Acc Diff: -3.57%, LR: 0.002510
2025-10-09 10:49:20,296 - CIFAR-100_Training - INFO - Starting Epoch 3/100
2025-10-09 10:50:16,640 - CIFAR-100_Training - INFO - Epoch  3: Train Loss: 2.9306, Train Acc: 26.24%, Test Loss: 2.7464, Test Acc: 29.52%, Acc Diff: -3.28%, LR: 0.002510
2025-10-09 10:50:16,640 - CIFAR-100_Training - INFO - Starting Epoch 4/100
2025-10-09 10:51:12,282 - CIFAR-100_Training - INFO - Epoch  4: Train Loss: 2.6260, Train Acc: 32.16%, Test Loss: 2.5548, Test Acc: 34.10%, Acc Diff: -1.94%, LR: 0.002510
2025-10-09 10:51:12,282 - CIFAR-100_Training - INFO - Starting Epoch 5/100
2025-10-09 10:52:09,881 - CIFAR-100_Training - INFO - Epoch  5: Train Loss: 2.4161, Train Acc: 36.32%, Test Loss: 2.3368, Test Acc: 38.27%, Acc Diff: -1.95%, LR: 0.002510
2025-10-09 10:52:09,882 - CIFAR-100_Training - INFO - Starting Epoch 6/100
2025-10-09 10:53:07,028 - CIFAR-100_Training - INFO - Epoch  6: Train Loss: 2.2642, Train Acc: 39.60%, Test Loss: 2.1890, Test Acc: 41.96%, Acc Diff: -2.36%, LR: 0.002510
2025-10-09 10:53:07,028 - CIFAR-100_Training - INFO - Starting Epoch 7/100
2025-10-09 10:54:02,502 - CIFAR-100_Training - INFO - Epoch  7: Train Loss: 2.1315, Train Acc: 42.71%, Test Loss: 2.1128, Test Acc: 42.94%, Acc Diff: -0.23%, LR: 0.002510
2025-10-09 10:54:02,502 - CIFAR-100_Training - INFO - Starting Epoch 8/100
2025-10-09 10:54:57,789 - CIFAR-100_Training - INFO - Epoch  8: Train Loss: 2.0369, Train Acc: 44.72%, Test Loss: 2.0696, Test Acc: 44.64%, Acc Diff: 0.08%, LR: 0.002510
2025-10-09 10:54:57,789 - CIFAR-100_Training - INFO - Starting Epoch 9/100
2025-10-09 10:55:53,244 - CIFAR-100_Training - INFO - Epoch  9: Train Loss: 1.9604, Train Acc: 46.46%, Test Loss: 2.1271, Test Acc: 43.84%, Acc Diff: 2.62%, LR: 0.002510
2025-10-09 10:55:53,244 - CIFAR-100_Training - INFO - Starting Epoch 10/100
2025-10-09 10:56:49,463 - CIFAR-100_Training - INFO - Epoch 10: Train Loss: 1.8823, Train Acc: 48.02%, Test Loss: 1.9379, Test Acc: 47.15%, Acc Diff: 0.87%, LR: 0.002510
2025-10-09 10:56:49,463 - CIFAR-100_Training - INFO - Starting Epoch 11/100
2025-10-09 10:57:45,172 - CIFAR-100_Training - INFO - Epoch 11: Train Loss: 1.8138, Train Acc: 49.74%, Test Loss: 1.8098, Test Acc: 50.43%, Acc Diff: -0.69%, LR: 0.002510
2025-10-09 10:57:45,172 - CIFAR-100_Training - INFO - Starting Epoch 12/100
2025-10-09 10:58:40,760 - CIFAR-100_Training - INFO - Epoch 12: Train Loss: 1.7549, Train Acc: 51.21%, Test Loss: 1.9104, Test Acc: 48.36%, Acc Diff: 2.85%, LR: 0.002510
2025-10-09 10:58:40,760 - CIFAR-100_Training - INFO - Starting Epoch 13/100
2025-10-09 10:59:36,572 - CIFAR-100_Training - INFO - Epoch 13: Train Loss: 1.7054, Train Acc: 52.26%, Test Loss: 1.7733, Test Acc: 51.39%, Acc Diff: 0.87%, LR: 0.002510
2025-10-09 10:59:36,572 - CIFAR-100_Training - INFO - Starting Epoch 14/100
2025-10-09 11:00:32,133 - CIFAR-100_Training - INFO - Epoch 14: Train Loss: 1.6524, Train Acc: 53.35%, Test Loss: 1.7744, Test Acc: 51.36%, Acc Diff: 1.99%, LR: 0.002510
2025-10-09 11:00:32,133 - CIFAR-100_Training - INFO - Starting Epoch 15/100
2025-10-09 11:01:27,645 - CIFAR-100_Training - INFO - Epoch 15: Train Loss: 1.6109, Train Acc: 54.98%, Test Loss: 1.7812, Test Acc: 51.73%, Acc Diff: 3.25%, LR: 0.002510
2025-10-09 11:01:27,645 - CIFAR-100_Training - INFO - Starting Epoch 16/100
2025-10-09 11:02:23,076 - CIFAR-100_Training - INFO - Epoch 16: Train Loss: 1.5662, Train Acc: 55.83%, Test Loss: 1.6517, Test Acc: 54.14%, Acc Diff: 1.69%, LR: 0.002510
2025-10-09 11:02:23,076 - CIFAR-100_Training - INFO - Starting Epoch 17/100
2025-10-09 11:03:18,682 - CIFAR-100_Training - INFO - Epoch 17: Train Loss: 1.5327, Train Acc: 56.91%, Test Loss: 1.7427, Test Acc: 53.13%, Acc Diff: 3.78%, LR: 0.002510
2025-10-09 11:03:18,682 - CIFAR-100_Training - INFO - Starting Epoch 18/100
2025-10-09 11:04:14,283 - CIFAR-100_Training - INFO - Epoch 18: Train Loss: 1.4999, Train Acc: 57.40%, Test Loss: 1.6383, Test Acc: 54.04%, Acc Diff: 3.36%, LR: 0.002510
2025-10-09 11:04:14,283 - CIFAR-100_Training - INFO - Starting Epoch 19/100
2025-10-09 11:05:09,852 - CIFAR-100_Training - INFO - Epoch 19: Train Loss: 1.4745, Train Acc: 58.19%, Test Loss: 1.6015, Test Acc: 55.82%, Acc Diff: 2.37%, LR: 0.002510
2025-10-09 11:05:09,852 - CIFAR-100_Training - INFO - Starting Epoch 20/100
2025-10-09 11:06:05,066 - CIFAR-100_Training - INFO - Epoch 20: Train Loss: 1.4341, Train Acc: 59.05%, Test Loss: 1.5808, Test Acc: 55.30%, Acc Diff: 3.75%, LR: 0.002510
2025-10-09 11:06:05,066 - CIFAR-100_Training - INFO - Starting Epoch 21/100
2025-10-09 11:07:00,985 - CIFAR-100_Training - INFO - Epoch 21: Train Loss: 1.4120, Train Acc: 59.35%, Test Loss: 1.6181, Test Acc: 55.29%, Acc Diff: 4.06%, LR: 0.002510
2025-10-09 11:07:00,985 - CIFAR-100_Training - INFO - Starting Epoch 22/100
2025-10-09 11:07:56,578 - CIFAR-100_Training - INFO - Epoch 22: Train Loss: 1.3808, Train Acc: 60.44%, Test Loss: 1.5498, Test Acc: 56.43%, Acc Diff: 4.01%, LR: 0.002510
2025-10-09 11:07:56,578 - CIFAR-100_Training - INFO - Starting Epoch 23/100
2025-10-09 11:08:51,792 - CIFAR-100_Training - INFO - Epoch 23: Train Loss: 1.3671, Train Acc: 60.55%, Test Loss: 1.5661, Test Acc: 56.06%, Acc Diff: 4.49%, LR: 0.002510
2025-10-09 11:08:51,792 - CIFAR-100_Training - INFO - Starting Epoch 24/100
2025-10-09 11:09:47,014 - CIFAR-100_Training - INFO - Epoch 24: Train Loss: 1.3436, Train Acc: 61.35%, Test Loss: 1.5373, Test Acc: 56.86%, Acc Diff: 4.49%, LR: 0.002510
2025-10-09 11:09:47,014 - CIFAR-100_Training - INFO - Starting Epoch 25/100
2025-10-09 11:10:42,381 - CIFAR-100_Training - INFO - Epoch 25: Train Loss: 1.3237, Train Acc: 61.60%, Test Loss: 1.5603, Test Acc: 56.51%, Acc Diff: 5.09%, LR: 0.002510
2025-10-09 11:10:42,381 - CIFAR-100_Training - INFO - Starting Epoch 26/100
2025-10-09 11:11:38,128 - CIFAR-100_Training - INFO - Epoch 26: Train Loss: 1.3098, Train Acc: 62.16%, Test Loss: 1.5362, Test Acc: 57.30%, Acc Diff: 4.86%, LR: 0.002510
2025-10-09 11:11:38,128 - CIFAR-100_Training - INFO - Starting Epoch 27/100
2025-10-09 11:12:34,105 - CIFAR-100_Training - INFO - Epoch 27: Train Loss: 1.2886, Train Acc: 62.70%, Test Loss: 1.5020, Test Acc: 58.43%, Acc Diff: 4.27%, LR: 0.002510
2025-10-09 11:12:34,105 - CIFAR-100_Training - INFO - Starting Epoch 28/100
2025-10-09 11:13:29,886 - CIFAR-100_Training - INFO - Epoch 28: Train Loss: 1.2776, Train Acc: 63.16%, Test Loss: 1.4823, Test Acc: 58.43%, Acc Diff: 4.73%, LR: 0.002510
2025-10-09 11:13:29,886 - CIFAR-100_Training - INFO - Starting Epoch 29/100
2025-10-09 11:14:25,849 - CIFAR-100_Training - INFO - Epoch 29: Train Loss: 1.2525, Train Acc: 63.68%, Test Loss: 1.5613, Test Acc: 57.35%, Acc Diff: 6.33%, LR: 0.002510
2025-10-09 11:14:25,849 - CIFAR-100_Training - INFO - Starting Epoch 30/100
2025-10-09 11:15:21,275 - CIFAR-100_Training - INFO - Epoch 30: Train Loss: 1.2474, Train Acc: 63.85%, Test Loss: 1.5143, Test Acc: 58.28%, Acc Diff: 5.57%, LR: 0.002510
2025-10-09 11:15:21,275 - CIFAR-100_Training - INFO - Starting Epoch 31/100
2025-10-09 11:16:16,805 - CIFAR-100_Training - INFO - Epoch 31: Train Loss: 1.2343, Train Acc: 64.27%, Test Loss: 1.4644, Test Acc: 59.21%, Acc Diff: 5.06%, LR: 0.002510
2025-10-09 11:16:16,805 - CIFAR-100_Training - INFO - Starting Epoch 32/100
2025-10-09 11:17:12,406 - CIFAR-100_Training - INFO - Epoch 32: Train Loss: 1.2139, Train Acc: 64.67%, Test Loss: 1.5031, Test Acc: 58.38%, Acc Diff: 6.29%, LR: 0.002510
2025-10-09 11:17:12,406 - CIFAR-100_Training - INFO - Starting Epoch 33/100
2025-10-09 11:18:07,807 - CIFAR-100_Training - INFO - Epoch 33: Train Loss: 1.2030, Train Acc: 64.91%, Test Loss: 1.4503, Test Acc: 59.76%, Acc Diff: 5.15%, LR: 0.002510
2025-10-09 11:18:07,807 - CIFAR-100_Training - INFO - Starting Epoch 34/100
2025-10-09 11:19:03,620 - CIFAR-100_Training - INFO - Epoch 34: Train Loss: 1.1960, Train Acc: 65.13%, Test Loss: 1.4621, Test Acc: 59.80%, Acc Diff: 5.33%, LR: 0.002510
2025-10-09 11:19:03,620 - CIFAR-100_Training - INFO - Starting Epoch 35/100
2025-10-09 11:19:59,191 - CIFAR-100_Training - INFO - Epoch 35: Train Loss: 1.1763, Train Acc: 65.54%, Test Loss: 1.4387, Test Acc: 60.17%, Acc Diff: 5.37%, LR: 0.002510
2025-10-09 11:19:59,191 - CIFAR-100_Training - INFO - Starting Epoch 36/100
2025-10-09 11:20:54,790 - CIFAR-100_Training - INFO - Epoch 36: Train Loss: 1.1767, Train Acc: 65.34%, Test Loss: 1.4205, Test Acc: 60.86%, Acc Diff: 4.48%, LR: 0.002510
2025-10-09 11:20:54,790 - CIFAR-100_Training - INFO - Starting Epoch 37/100
2025-10-09 11:21:50,138 - CIFAR-100_Training - INFO - Epoch 37: Train Loss: 1.1550, Train Acc: 66.12%, Test Loss: 1.4405, Test Acc: 59.92%, Acc Diff: 6.20%, LR: 0.002510
2025-10-09 11:21:50,138 - CIFAR-100_Training - INFO - Starting Epoch 38/100
2025-10-09 11:22:45,636 - CIFAR-100_Training - INFO - Epoch 38: Train Loss: 1.1553, Train Acc: 66.27%, Test Loss: 1.4466, Test Acc: 59.57%, Acc Diff: 6.70%, LR: 0.002510
2025-10-09 11:22:45,636 - CIFAR-100_Training - INFO - Starting Epoch 39/100
2025-10-09 11:23:41,281 - CIFAR-100_Training - INFO - Epoch 39: Train Loss: 1.1462, Train Acc: 66.28%, Test Loss: 1.4187, Test Acc: 60.46%, Acc Diff: 5.82%, LR: 0.002510
2025-10-09 11:23:41,281 - CIFAR-100_Training - INFO - Starting Epoch 40/100
2025-10-09 11:24:36,717 - CIFAR-100_Training - INFO - Epoch 40: Train Loss: 1.1337, Train Acc: 66.56%, Test Loss: 1.4134, Test Acc: 60.71%, Acc Diff: 5.85%, LR: 0.002510
2025-10-09 11:24:36,717 - CIFAR-100_Training - INFO - Starting Epoch 41/100
2025-10-09 11:25:32,252 - CIFAR-100_Training - INFO - Epoch 41: Train Loss: 1.1231, Train Acc: 66.83%, Test Loss: 1.4681, Test Acc: 59.95%, Acc Diff: 6.88%, LR: 0.002510
2025-10-09 11:25:32,252 - CIFAR-100_Training - INFO - Starting Epoch 42/100
2025-10-09 11:26:28,472 - CIFAR-100_Training - INFO - Epoch 42: Train Loss: 1.1159, Train Acc: 66.83%, Test Loss: 1.4572, Test Acc: 59.81%, Acc Diff: 7.02%, LR: 0.002510
2025-10-09 11:26:28,472 - CIFAR-100_Training - INFO - Starting Epoch 43/100
2025-10-09 11:27:24,152 - CIFAR-100_Training - INFO - Epoch 43: Train Loss: 1.1070, Train Acc: 67.29%, Test Loss: 1.4550, Test Acc: 60.29%, Acc Diff: 7.00%, LR: 0.002510
2025-10-09 11:27:24,152 - CIFAR-100_Training - INFO - Starting Epoch 44/100
2025-10-09 11:28:19,845 - CIFAR-100_Training - INFO - Epoch 44: Train Loss: 1.1056, Train Acc: 67.44%, Test Loss: 1.4253, Test Acc: 61.11%, Acc Diff: 6.33%, LR: 0.002510
2025-10-09 11:28:19,845 - CIFAR-100_Training - INFO - Starting Epoch 45/100
2025-10-09 11:29:15,284 - CIFAR-100_Training - INFO - Epoch 45: Train Loss: 1.0996, Train Acc: 67.52%, Test Loss: 1.4173, Test Acc: 60.79%, Acc Diff: 6.73%, LR: 0.002510
2025-10-09 11:29:15,284 - CIFAR-100_Training - INFO - Starting Epoch 46/100
2025-10-09 11:30:10,405 - CIFAR-100_Training - INFO - Epoch 46: Train Loss: 1.0771, Train Acc: 68.52%, Test Loss: 1.4680, Test Acc: 60.12%, Acc Diff: 8.40%, LR: 0.002510
2025-10-09 11:30:10,405 - CIFAR-100_Training - INFO - Starting Epoch 47/100
2025-10-09 11:31:05,965 - CIFAR-100_Training - INFO - Epoch 47: Train Loss: 1.0853, Train Acc: 67.78%, Test Loss: 1.4765, Test Acc: 59.81%, Acc Diff: 7.97%, LR: 0.002510
2025-10-09 11:31:05,965 - CIFAR-100_Training - INFO - Starting Epoch 48/100
2025-10-09 11:32:01,069 - CIFAR-100_Training - INFO - Epoch 48: Train Loss: 1.0669, Train Acc: 68.16%, Test Loss: 1.4036, Test Acc: 61.42%, Acc Diff: 6.74%, LR: 0.002510
2025-10-09 11:32:01,069 - CIFAR-100_Training - INFO - Starting Epoch 49/100
2025-10-09 11:32:56,300 - CIFAR-100_Training - INFO - Epoch 49: Train Loss: 1.0673, Train Acc: 68.22%, Test Loss: 1.4234, Test Acc: 60.84%, Acc Diff: 7.38%, LR: 0.002510
2025-10-09 11:32:56,300 - CIFAR-100_Training - INFO - Starting Epoch 50/100
2025-10-09 11:33:51,678 - CIFAR-100_Training - INFO - Epoch 50: Train Loss: 1.0580, Train Acc: 68.66%, Test Loss: 1.5015, Test Acc: 59.44%, Acc Diff: 9.22%, LR: 0.002510
2025-10-09 11:33:51,678 - CIFAR-100_Training - INFO - Starting Epoch 51/100
2025-10-09 11:34:47,453 - CIFAR-100_Training - INFO - Epoch 51: Train Loss: 1.0602, Train Acc: 68.55%, Test Loss: 1.3636, Test Acc: 62.26%, Acc Diff: 6.29%, LR: 0.002510
2025-10-09 11:34:47,453 - CIFAR-100_Training - INFO - Starting Epoch 52/100
2025-10-09 11:35:43,126 - CIFAR-100_Training - INFO - Epoch 52: Train Loss: 1.0528, Train Acc: 68.85%, Test Loss: 1.3901, Test Acc: 61.21%, Acc Diff: 7.64%, LR: 0.002510
2025-10-09 11:35:43,126 - CIFAR-100_Training - INFO - Starting Epoch 53/100
2025-10-09 11:36:38,843 - CIFAR-100_Training - INFO - Epoch 53: Train Loss: 1.0382, Train Acc: 69.09%, Test Loss: 1.4177, Test Acc: 60.71%, Acc Diff: 8.38%, LR: 0.002510
2025-10-09 11:36:38,843 - CIFAR-100_Training - INFO - Starting Epoch 54/100
2025-10-09 11:37:34,443 - CIFAR-100_Training - INFO - Epoch 54: Train Loss: 1.0419, Train Acc: 69.06%, Test Loss: 1.4514, Test Acc: 60.14%, Acc Diff: 8.92%, LR: 0.002510
2025-10-09 11:37:34,443 - CIFAR-100_Training - INFO - Starting Epoch 55/100
2025-10-09 11:38:29,774 - CIFAR-100_Training - INFO - Epoch 55: Train Loss: 1.0371, Train Acc: 69.36%, Test Loss: 1.4218, Test Acc: 60.96%, Acc Diff: 8.40%, LR: 0.002510
2025-10-09 11:38:29,774 - CIFAR-100_Training - INFO - Starting Epoch 56/100
2025-10-09 11:39:25,038 - CIFAR-100_Training - INFO - Epoch 56: Train Loss: 1.0290, Train Acc: 69.35%, Test Loss: 1.4114, Test Acc: 60.83%, Acc Diff: 8.52%, LR: 0.002510
2025-10-09 11:39:25,038 - CIFAR-100_Training - INFO - Starting Epoch 57/100
2025-10-09 11:40:20,538 - CIFAR-100_Training - INFO - Epoch 57: Train Loss: 1.0224, Train Acc: 69.56%, Test Loss: 1.3999, Test Acc: 61.51%, Acc Diff: 8.05%, LR: 0.002510
2025-10-09 11:40:20,538 - CIFAR-100_Training - INFO - Starting Epoch 58/100
2025-10-09 11:41:16,599 - CIFAR-100_Training - INFO - Epoch 58: Train Loss: 1.0201, Train Acc: 69.89%, Test Loss: 1.4246, Test Acc: 61.19%, Acc Diff: 8.70%, LR: 0.002510
2025-10-09 11:41:16,599 - CIFAR-100_Training - INFO - Starting Epoch 59/100
2025-10-09 11:42:11,873 - CIFAR-100_Training - INFO - Epoch 59: Train Loss: 1.0135, Train Acc: 69.87%, Test Loss: 1.3710, Test Acc: 61.96%, Acc Diff: 7.91%, LR: 0.002510
2025-10-09 11:42:11,873 - CIFAR-100_Training - INFO - Starting Epoch 60/100
2025-10-09 11:43:07,470 - CIFAR-100_Training - INFO - Epoch 60: Train Loss: 1.0089, Train Acc: 69.83%, Test Loss: 1.3667, Test Acc: 62.97%, Acc Diff: 6.86%, LR: 0.002510
2025-10-09 11:43:07,470 - CIFAR-100_Training - INFO - Starting Epoch 61/100
2025-10-09 11:44:03,153 - CIFAR-100_Training - INFO - Epoch 61: Train Loss: 1.0004, Train Acc: 70.46%, Test Loss: 1.4088, Test Acc: 61.85%, Acc Diff: 8.61%, LR: 0.002510
2025-10-09 11:44:03,153 - CIFAR-100_Training - INFO - Starting Epoch 62/100
2025-10-09 11:44:58,667 - CIFAR-100_Training - INFO - Epoch 62: Train Loss: 1.0043, Train Acc: 70.15%, Test Loss: 1.3723, Test Acc: 62.22%, Acc Diff: 7.93%, LR: 0.001255
2025-10-09 11:44:58,667 - CIFAR-100_Training - INFO - Starting Epoch 63/100
2025-10-09 11:45:54,259 - CIFAR-100_Training - INFO - Epoch 63: Train Loss: 0.8096, Train Acc: 75.94%, Test Loss: 1.1996, Test Acc: 66.53%, Acc Diff: 9.41%, LR: 0.001255
2025-10-09 11:45:54,259 - CIFAR-100_Training - INFO - Starting Epoch 64/100
2025-10-09 11:46:50,069 - CIFAR-100_Training - INFO - Epoch 64: Train Loss: 0.7566, Train Acc: 77.15%, Test Loss: 1.2197, Test Acc: 66.32%, Acc Diff: 10.83%, LR: 0.001255
2025-10-09 11:46:50,069 - CIFAR-100_Training - INFO - Starting Epoch 65/100
2025-10-09 11:47:45,817 - CIFAR-100_Training - INFO - Epoch 65: Train Loss: 0.7475, Train Acc: 77.28%, Test Loss: 1.2319, Test Acc: 66.38%, Acc Diff: 10.90%, LR: 0.001255
2025-10-09 11:47:45,817 - CIFAR-100_Training - INFO - Starting Epoch 66/100
2025-10-09 11:48:40,982 - CIFAR-100_Training - INFO - Epoch 66: Train Loss: 0.7361, Train Acc: 77.87%, Test Loss: 1.2430, Test Acc: 66.14%, Acc Diff: 11.73%, LR: 0.001255
2025-10-09 11:48:40,982 - CIFAR-100_Training - INFO - Starting Epoch 67/100
2025-10-09 11:49:36,798 - CIFAR-100_Training - INFO - Epoch 67: Train Loss: 0.7276, Train Acc: 77.98%, Test Loss: 1.2664, Test Acc: 65.58%, Acc Diff: 12.40%, LR: 0.001255
2025-10-09 11:49:36,798 - CIFAR-100_Training - INFO - Starting Epoch 68/100
2025-10-09 11:50:32,384 - CIFAR-100_Training - INFO - Epoch 68: Train Loss: 0.7142, Train Acc: 78.24%, Test Loss: 1.2874, Test Acc: 65.39%, Acc Diff: 12.85%, LR: 0.001255
2025-10-09 11:50:32,384 - CIFAR-100_Training - INFO - Starting Epoch 69/100
2025-10-09 11:51:28,372 - CIFAR-100_Training - INFO - Epoch 69: Train Loss: 0.7081, Train Acc: 78.44%, Test Loss: 1.2663, Test Acc: 65.89%, Acc Diff: 12.55%, LR: 0.001255
2025-10-09 11:51:28,372 - CIFAR-100_Training - INFO - Starting Epoch 70/100
2025-10-09 11:52:23,917 - CIFAR-100_Training - INFO - Epoch 70: Train Loss: 0.7067, Train Acc: 78.41%, Test Loss: 1.2514, Test Acc: 65.94%, Acc Diff: 12.47%, LR: 0.001255
2025-10-09 11:52:23,917 - CIFAR-100_Training - INFO - Starting Epoch 71/100
2025-10-09 11:53:19,535 - CIFAR-100_Training - INFO - Epoch 71: Train Loss: 0.6952, Train Acc: 78.90%, Test Loss: 1.2874, Test Acc: 65.20%, Acc Diff: 13.70%, LR: 0.001255
2025-10-09 11:53:19,535 - CIFAR-100_Training - INFO - Starting Epoch 72/100
2025-10-09 11:54:15,281 - CIFAR-100_Training - INFO - Epoch 72: Train Loss: 0.6988, Train Acc: 78.69%, Test Loss: 1.2785, Test Acc: 65.58%, Acc Diff: 13.11%, LR: 0.001255
2025-10-09 11:54:15,281 - CIFAR-100_Training - INFO - Starting Epoch 73/100
2025-10-09 11:55:10,917 - CIFAR-100_Training - INFO - Epoch 73: Train Loss: 0.6936, Train Acc: 78.87%, Test Loss: 1.2565, Test Acc: 66.27%, Acc Diff: 12.60%, LR: 0.001255
2025-10-09 11:55:10,917 - CIFAR-100_Training - INFO - Starting Epoch 74/100
2025-10-09 11:56:06,549 - CIFAR-100_Training - INFO - Epoch 74: Train Loss: 0.6850, Train Acc: 78.99%, Test Loss: 1.2495, Test Acc: 66.51%, Acc Diff: 12.48%, LR: 0.000628
2025-10-09 11:56:06,549 - CIFAR-100_Training - INFO - Starting Epoch 75/100
2025-10-09 11:57:02,038 - CIFAR-100_Training - INFO - Epoch 75: Train Loss: 0.5593, Train Acc: 82.89%, Test Loss: 1.1968, Test Acc: 67.78%, Acc Diff: 15.11%, LR: 0.000628 (OVERFITTING: 1 epochs)
2025-10-09 11:57:02,038 - CIFAR-100_Training - INFO - Starting Epoch 76/100
2025-10-09 11:57:57,478 - CIFAR-100_Training - INFO - Epoch 76: Train Loss: 0.5267, Train Acc: 84.00%, Test Loss: 1.1882, Test Acc: 68.21%, Acc Diff: 15.79%, LR: 0.000628 (OVERFITTING: 2 epochs)
2025-10-09 11:57:57,478 - CIFAR-100_Training - INFO - Starting Epoch 77/100
2025-10-09 11:58:52,847 - CIFAR-100_Training - INFO - Epoch 77: Train Loss: 0.5174, Train Acc: 84.23%, Test Loss: 1.1952, Test Acc: 68.42%, Acc Diff: 15.81%, LR: 0.000628 (OVERFITTING: 3 epochs)
2025-10-09 11:58:52,863 - CIFAR-100_Training - INFO - Starting Epoch 78/100
2025-10-09 11:59:48,548 - CIFAR-100_Training - INFO - Epoch 78: Train Loss: 0.5044, Train Acc: 84.62%, Test Loss: 1.2101, Test Acc: 67.91%, Acc Diff: 16.71%, LR: 0.000628 (OVERFITTING: 4 epochs)
2025-10-09 11:59:48,548 - CIFAR-100_Training - INFO - Starting Epoch 79/100
2025-10-09 12:00:44,152 - CIFAR-100_Training - INFO - Epoch 79: Train Loss: 0.5019, Train Acc: 84.70%, Test Loss: 1.2260, Test Acc: 67.84%, Acc Diff: 16.86%, LR: 0.000628 (OVERFITTING: 5 epochs)
2025-10-09 12:00:44,152 - CIFAR-100_Training - INFO - Starting Epoch 80/100
2025-10-09 12:01:39,659 - CIFAR-100_Training - INFO - Epoch 80: Train Loss: 0.4962, Train Acc: 84.67%, Test Loss: 1.2134, Test Acc: 67.81%, Acc Diff: 16.86%, LR: 0.000628 (OVERFITTING: 6 epochs)
2025-10-09 12:01:39,659 - CIFAR-100_Training - INFO - Starting Epoch 81/100
2025-10-09 12:02:35,094 - CIFAR-100_Training - INFO - Epoch 81: Train Loss: 0.4901, Train Acc: 84.99%, Test Loss: 1.2278, Test Acc: 68.07%, Acc Diff: 16.92%, LR: 0.000628 (OVERFITTING: 7 epochs)
2025-10-09 12:02:35,094 - CIFAR-100_Training - INFO - Starting Epoch 82/100
2025-10-09 12:03:30,623 - CIFAR-100_Training - INFO - Epoch 82: Train Loss: 0.4804, Train Acc: 85.39%, Test Loss: 1.2550, Test Acc: 67.80%, Acc Diff: 17.59%, LR: 0.000628 (OVERFITTING: 8 epochs)
2025-10-09 12:03:30,623 - CIFAR-100_Training - INFO - Starting Epoch 83/100
2025-10-09 12:04:26,397 - CIFAR-100_Training - INFO - Epoch 83: Train Loss: 0.4829, Train Acc: 85.22%, Test Loss: 1.2429, Test Acc: 67.58%, Acc Diff: 17.64%, LR: 0.000628 (OVERFITTING: 9 epochs)
2025-10-09 12:04:26,397 - CIFAR-100_Training - INFO - Starting Epoch 84/100
2025-10-09 12:05:22,140 - CIFAR-100_Training - INFO - Epoch 84: Train Loss: 0.4765, Train Acc: 85.37%, Test Loss: 1.2527, Test Acc: 67.64%, Acc Diff: 17.73%, LR: 0.000628 (OVERFITTING: 10 epochs)
2025-10-09 12:05:22,140 - CIFAR-100_Training - INFO - Starting Epoch 85/100
2025-10-09 12:06:18,245 - CIFAR-100_Training - INFO - Epoch 85: Train Loss: 0.4702, Train Acc: 85.63%, Test Loss: 1.2512, Test Acc: 67.49%, Acc Diff: 18.14%, LR: 0.000628 (OVERFITTING: 11 epochs)
2025-10-09 12:06:18,245 - CIFAR-100_Training - INFO - Starting Epoch 86/100
2025-10-09 12:07:13,858 - CIFAR-100_Training - INFO - Epoch 86: Train Loss: 0.4721, Train Acc: 85.36%, Test Loss: 1.2383, Test Acc: 67.86%, Acc Diff: 17.50%, LR: 0.000628 (OVERFITTING: 12 epochs)
2025-10-09 12:07:13,858 - CIFAR-100_Training - INFO - Starting Epoch 87/100
2025-10-09 12:08:09,377 - CIFAR-100_Training - INFO - Epoch 87: Train Loss: 0.4660, Train Acc: 85.64%, Test Loss: 1.2472, Test Acc: 67.61%, Acc Diff: 18.03%, LR: 0.000314 (OVERFITTING: 13 epochs)
2025-10-09 12:08:09,377 - CIFAR-100_Training - INFO - Starting Epoch 88/100
2025-10-09 12:09:04,613 - CIFAR-100_Training - INFO - Epoch 88: Train Loss: 0.4073, Train Acc: 87.84%, Test Loss: 1.2127, Test Acc: 68.52%, Acc Diff: 19.32%, LR: 0.000314 (OVERFITTING: 14 epochs)
2025-10-09 12:09:04,613 - CIFAR-100_Training - INFO - Starting Epoch 89/100
2025-10-09 12:10:00,193 - CIFAR-100_Training - INFO - Epoch 89: Train Loss: 0.3843, Train Acc: 88.55%, Test Loss: 1.2028, Test Acc: 69.03%, Acc Diff: 19.52%, LR: 0.000314 (OVERFITTING: 15 epochs)
2025-10-09 12:10:00,193 - CIFAR-100_Training - INFO - Starting Epoch 90/100
2025-10-09 12:10:56,019 - CIFAR-100_Training - INFO - Epoch 90: Train Loss: 0.3766, Train Acc: 88.60%, Test Loss: 1.2124, Test Acc: 69.03%, Acc Diff: 19.57%, LR: 0.000314 (OVERFITTING: 16 epochs)
2025-10-09 12:10:56,019 - CIFAR-100_Training - INFO - Starting Epoch 91/100
2025-10-09 12:11:51,806 - CIFAR-100_Training - INFO - Epoch 91: Train Loss: 0.3739, Train Acc: 88.81%, Test Loss: 1.2196, Test Acc: 68.59%, Acc Diff: 20.22%, LR: 0.000314 (OVERFITTING: 17 epochs)
2025-10-09 12:11:51,806 - CIFAR-100_Training - INFO - Starting Epoch 92/100
2025-10-09 12:12:47,710 - CIFAR-100_Training - INFO - Epoch 92: Train Loss: 0.3677, Train Acc: 88.95%, Test Loss: 1.2321, Test Acc: 68.64%, Acc Diff: 20.31%, LR: 0.000314 (OVERFITTING: 18 epochs)
2025-10-09 12:12:47,710 - CIFAR-100_Training - INFO - Starting Epoch 93/100
2025-10-09 12:13:42,900 - CIFAR-100_Training - INFO - Epoch 93: Train Loss: 0.3607, Train Acc: 89.07%, Test Loss: 1.2488, Test Acc: 68.52%, Acc Diff: 20.55%, LR: 0.000314 (OVERFITTING: 19 epochs)
2025-10-09 12:13:42,900 - CIFAR-100_Training - INFO - Starting Epoch 94/100
2025-10-09 12:14:38,354 - CIFAR-100_Training - INFO - Epoch 94: Train Loss: 0.3582, Train Acc: 89.41%, Test Loss: 1.2445, Test Acc: 68.73%, Acc Diff: 20.68%, LR: 0.000314 (OVERFITTING: 20 epochs)
2025-10-09 12:14:38,354 - CIFAR-100_Training - INFO - Starting Epoch 95/100
2025-10-09 12:15:34,494 - CIFAR-100_Training - INFO - Epoch 95: Train Loss: 0.3548, Train Acc: 89.29%, Test Loss: 1.2457, Test Acc: 68.73%, Acc Diff: 20.56%, LR: 0.000314 (OVERFITTING: 21 epochs)
2025-10-09 12:15:34,494 - CIFAR-100_Training - INFO - Starting Epoch 96/100
2025-10-09 12:16:30,726 - CIFAR-100_Training - INFO - Epoch 96: Train Loss: 0.3594, Train Acc: 89.11%, Test Loss: 1.2378, Test Acc: 68.72%, Acc Diff: 20.39%, LR: 0.000314 (OVERFITTING: 22 epochs)
2025-10-09 12:16:30,726 - CIFAR-100_Training - INFO - Starting Epoch 97/100
2025-10-09 12:17:26,152 - CIFAR-100_Training - INFO - Epoch 97: Train Loss: 0.3525, Train Acc: 89.30%, Test Loss: 1.2397, Test Acc: 68.80%, Acc Diff: 20.50%, LR: 0.000314 (OVERFITTING: 23 epochs)
2025-10-09 12:17:26,152 - CIFAR-100_Training - INFO - Starting Epoch 98/100
2025-10-09 12:18:21,426 - CIFAR-100_Training - INFO - Epoch 98: Train Loss: 0.3523, Train Acc: 89.44%, Test Loss: 1.2501, Test Acc: 68.72%, Acc Diff: 20.72%, LR: 0.000157 (OVERFITTING: 24 epochs)
2025-10-09 12:18:21,426 - CIFAR-100_Training - INFO - Starting Epoch 99/100
2025-10-09 12:19:16,864 - CIFAR-100_Training - INFO - Epoch 99: Train Loss: 0.3198, Train Acc: 90.47%, Test Loss: 1.2240, Test Acc: 69.28%, Acc Diff: 21.19%, LR: 0.000157 (OVERFITTING: 25 epochs)
2025-10-09 12:19:16,864 - CIFAR-100_Training - INFO - Starting Epoch 100/100
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Epoch 100: Train Loss: 0.3142, Train Acc: 90.71%, Test Loss: 1.2220, Test Acc: 69.24%, Acc Diff: 21.47%, LR: 0.000157 (OVERFITTING: 26 epochs)
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Training completed!
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Final Results: {'final_train_loss': 0.31422219907536225, 'final_test_loss': 1.2219862369537353, 'final_train_accuracy': 90.708, 'final_test_accuracy': 69.24, 'best_test_accuracy': 69.28, 'final_accuracy_difference': 21.468000000000004, 'max_accuracy_difference': 21.468000000000004, 'avg_accuracy_difference': 9.132480000000001, 'overfitting_epochs': 26, 'stopped_due_to_overfitting': True}
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - OVERFITTING ANALYSIS
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Final accuracy difference: 21.47%
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Maximum accuracy difference: 21.47%
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Average accuracy difference: 9.13%
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Consecutive overfitting epochs: 26
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - Stopped due to overfitting: True
2025-10-09 12:20:12,149 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 12:20:15,120 - CIFAR-100_Training - INFO - Training curves saved to: C:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS8\Model_Evolution\FineTune\logs\training_curves_20251009_122012.png
2025-10-09 19:41:12,021 - CIFAR-100_Training - INFO - Model saved to: C:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERAS8\Model_Evolution\FineTune\models\cifar100_model_20251009_194111.pth
2025-10-09 19:41:12,022 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 19:41:12,022 - CIFAR-100_Training - INFO - TRAINING PIPELINE COMPLETED SUCCESSFULLY
2025-10-09 19:41:12,022 - CIFAR-100_Training - INFO - ==================================================
2025-10-09 19:41:12,023 - CIFAR-100_Training - INFO - Final Metrics: {'final_train_loss': 0.31422219907536225, 'final_test_loss': 1.2219862369537353, 'final_train_accuracy': 90.708, 'final_test_accuracy': 69.24, 'best_test_accuracy': 69.28, 'final_accuracy_difference': 21.468000000000004, 'max_accuracy_difference': 21.468000000000004, 'avg_accuracy_difference': 9.132480000000001, 'overfitting_epochs': 26, 'stopped_due_to_overfitting': True}
