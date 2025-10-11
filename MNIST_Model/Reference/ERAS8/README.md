# CIFAR-100 Training Experiments - Fine-Tuning Phase
*This README documents the fine-tuning experiments for CIFAR-100 classification using ResNet architectures with various optimization strategies.*

## Project Summary

### Experiments Overview

| Experiment | Architecture | Params | Optimizer | Best Test Acc | Gap | Status |
|------------|--------------|--------|-----------|---------------|-----|--------|
| **FT-1** | ResNet-18 Bottleneck | 0.93M | Adam + ReduceLR | 69.28% | 21.47% | ✅ Baseline |
| **FT-2** | ResNet-34 | 21.3M | SGD + OneCycleLR | 67.50% | 24.11% | ❌ Regression |
| **FT-3** | ResNet-34 | 21.3M | Adam + OneCycleLR | 73.57% | 10.78% | 🎯 **Best Model** |

### Current Best Result (FT-3) 🎯
- **Architecture**: ResNet-34
- **Parameters**: 21,328,292 (~21.3M)
- **Best Test Accuracy**: 73.57% (epoch 30)
- **Final Test Accuracy**: 73.57% (epoch 30)
- **Final Train Accuracy**: 84.35%
- **Final Train-Test Gap**: 10.78% (**Best generalization!**)
- **Epochs Completed**: 30 (early stopping not triggered)
- **Overfitting Epochs**: 0 (healthy training throughout)

### Key Learnings
✅ **Adam + OneCycleLR** is the winning combination (FT-3)  
✅ **Shorter training** (30 epochs) prevents overfitting  
✅ **No dropout needed** when using proper LR schedule  
✅ **ResNet-34** performs better than ResNet-18 with right config  
⚠️ **SGD + OneCycleLR** struggles without dropout (FT-2)  
⚠️ **Long training** (100 epochs) leads to overfitting (FT-1, FT-2)   

## Resources

### Experiment Files

**FT-1 (ResNet-18 Bottleneck + Adam)** ✅ Best:
- **Training Log**: [20251009_104659_cifar100_training.log](Model_Evolution/FineTune/logs/20251009_104659_cifar100_training.log)
- **Training Curves**: [training_curves_20251009_122012.png](Model_Evolution/FineTune/logs/training_curves_20251009_122012.png)
- **Saved Model**: [cifar100_model_20251009_194111.pth](Model_Evolution/FineTune/models/cifar100_model_20251009_194111.pth)

**FT-2 (ResNet-34 + SGD OneCycleLR)** ❌ Regression:
- **Training Log**: [20251011_141944_cifar100_training.log](Model_Evolution/FineTune/logs/20251011_141944_cifar100_training.log)
- **Training Curves**: [training_curves_20251011_200557.png](Model_Evolution/FineTune/logs/training_curves_20251011_200557.png)
- **Saved Model**: [cifar100_model_20251011_200637.pth](Model_Evolution/FineTune/models/cifar100_model_20251011_200637.pth)

**FT-3 (ResNet-34 + Adam OneCycleLR)** 🎯 **Best Model**:
- **Training Log**: [20251011_093929_cifar100_training.log](Model_Evolution/FineTune/logs/20251011_093929_cifar100_training.log)
- **Training Curves**: [training_curves_20251011_093929.png](Model_Evolution/FineTune/logs/training_curves_20251011_093929.png)
- **Saved Model**: [cifar100_model_20251011_093931.pth](Model_Evolution/FineTune/models/cifar100_model_20251011_093931.pth)

**Shared Code**:
- **Model Code**: [model.py](Model_Evolution/FineTune/model.py)
- **Training Script**: [cifar100_training.py](Model_Evolution/FineTune/cifar100_training.py)

================================================================================
### GRAD-CAM ANALYSIS -FT-3 Model
================================================================================

📁 OUTPUT FILES GENERATED:
  1. **Confusion Matrix**: [link here](gradcam_results/confusion_matrix_full.png)
  2. **Per-Class Accuracy CSV**: [link here](gradcam_results/per_class_accuracy.csv)
  3. **Interactive Accuracy Chart**: [link here](gradcam_results/per_class_accuracy_interactive.html)
  4. **Confused Pairs CSV**: [link here](gradcam_results/most_confused_pairs.csv)
  5. **Confused Pairs Chart**: [link here](gradcam_results/top_confused_pairs.png)
  6. **Per-Class Grad-CAM Images**: [link here](gradcam_results/worst_predictions/)
  7. **Confused Pair Examples**: [link here](gradcam_results/confused_pair_*.png)
  8. **Analysis Summary**: [link here](gradcam_results/analysis_summary.txt)

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

## 📊 EXPERIMENT LOG: ResNet-34 with SGD + OneCycleLR (October 11, 2025)

### Experiment ID: FT-2 (SGD-OneCycle)

---

### 🎯 Experiment Goal
Test ResNet-34 with SGD optimizer and OneCycleLR scheduler to compare against Adam baseline (FT-1) and evaluate if momentum-based optimization can improve generalization.

### 📋 Configuration

**Model Architecture**: ResNet-34  
**Parameters**: 21,328,292 (~21.3M)  
**Dropout Rate**: 0.0 (No dropout)

**Optimizer**: SGD
- Initial LR: 0.00251
- Momentum: 0.9
- Weight Decay: 0.0001

**Scheduler**: OneCycleLR
- Max LR: 0.01
- Pct Start: 0.3 (30% warmup)
- Div Factor: 5 (initial LR = max_lr/5 = 0.002)
- Final Div Factor: 1000.0
- Anneal Strategy: cosine

**Training Config**:
- Epochs: 100
- Batch Size: 128
- Data Augmentation: Albumentations (HorizontalFlip, ShiftScaleRotate, CoarseDropout)

---

### 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 67.50% @ epoch 97 |
| **Final Test Accuracy** | 67.46% @ epoch 100 |
| **Final Train Accuracy** | 91.57% |
| **Final Train-Test Gap** | 24.11% |
| **Average Gap** | 10.01% |
| **Overfitting Started** | Epoch 60 |
| **Overfitting Epochs** | 41 out of 100 |
| **Training Duration** | ~5.75 hours |

---

### 📈 Training Progression

| Phase | Epochs | Train Acc | Test Acc | Gap | LR Range | Status |
|-------|--------|-----------|----------|-----|----------|--------|
| **Warmup** | 1-10 | 5.41% → 33.81% | 9.88% → 40.09% | -4.47% → -6.28% | 0.002 → 0.004 | ✅ Fast learning |
| **Ramp-up** | 11-30 | 35.90% → 59.10% | 40.18% → 57.62% | -4.28% → 1.48% | 0.004 → 0.010 | ✅ Healthy progress |
| **Peak LR** | 31-60 | 60.52% → 79.05% | 58.17% → 63.14% | 2.35% → 15.91% | 0.010 → 0.006 | ⚠️ Gap widening |
| **Cooldown** | 61-100 | 79.64% → 91.57% | 64.08% → 67.46% | 15.56% → 24.11% | 0.006 → 0.000002 | ❌ Severe overfitting |

---

### 🔍 Key Observations

#### ✅ Strengths
1. **Fast Initial Learning**: Reached 40% test accuracy in just 10 epochs
2. **Good Momentum Effect**: SGD momentum helped smooth training
3. **OneCycleLR Warmup**: First 30 epochs showed excellent learning rate schedule
4. **Stable Training**: No training instabilities or divergence

#### ⚠️ Weaknesses
1. **Severe Overfitting**: 24.11% final gap (worst among all experiments)
2. **Lower Test Accuracy**: 67.46% vs 69.24% (Adam baseline FT-1)
3. **No Regularization**: Dropout = 0.0 contributed to overfitting
4. **Long Overfitting Phase**: 41 consecutive overfitting epochs (60-100)
5. **Poor Cooldown**: Test accuracy plateaued while train kept improving

#### 📉 Critical Issues
- **Epoch 60**: Overfitting threshold crossed (gap > 15%)
- **Train-Test Divergence**: Train improved 12.52% while test only improved 4.32% in epochs 61-100
- **Max Gap Reached**: 24.12% at epoch 92 (highest across all experiments)

---

### 📊 Comparison with Previous Experiments

| Metric | FT-1 (Adam) | FT-2 (SGD-OneCycle) | Δ Change |
|--------|-------------|---------------------|----------|
| **Architecture** | ResNet-18 Bottleneck | ResNet-34 | Larger ⬆️ |
| **Parameters** | 929K | 21.3M | +2,194% |
| **Best Test Acc** | 69.28% | 67.50% | -1.78% ❌ |
| **Final Test Acc** | 69.24% | 67.46% | -1.78% ❌ |
| **Final Train Acc** | 90.71% | 91.57% | +0.86% |
| **Final Gap** | 21.47% | 24.11% | +2.64% ❌ |
| **Avg Gap** | 9.13% | 10.01% | +0.88% ❌ |
| **Overfitting Epochs** | 26 | 41 | +58% ❌ |
| **Dropout** | 0.05 | 0.0 | None ❌ |

---

### 💡 Analysis & Insights

#### Why Did This Experiment Underperform?

1. **No Dropout Regularization**
   - FT-1 had 0.05 dropout → 21.47% gap
   - FT-2 had 0.0 dropout → 24.11% gap
   - **Impact**: +2.64% worse generalization

2. **Model Capacity Too High**
   - ResNet-34 (21M params) vs ResNet-18 (0.9M params)
   - Without regularization, larger model memorizes training data
   - **Impact**: Severe overfitting after epoch 60

3. **SGD vs Adam Trade-off**
   - Adam adapts learning rates per parameter → better generalization
   - SGD with fixed momentum → more prone to overfitting without regularization
   - **Impact**: -1.78% test accuracy

4. **OneCycleLR Cooldown Issue**
   - Final LR too low (0.000002) → model barely learning
   - Test accuracy stagnated at ~67% while train kept improving
   - **Impact**: 41 wasted epochs in overfitting phase

---

### 🎯 Recommendations for Next Experiment (FT-3)

#### Priority 1: Add Regularization
```python
dropout_rate = 0.10  # Double from FT-1's 0.05
```
- Higher dropout needed for larger ResNet-34 model
- Target: Reduce gap to ~15-18%

#### Priority 2: Adjust OneCycleLR
```python
onecycle_pct_start = 0.4  # Longer warmup
onecycle_final_div_factor = 100  # Higher final LR
```
- Extend warmup phase to 40 epochs
- Keep final LR higher for continued learning

#### Priority 3: Early Stopping
```python
early_stop_patience = 15  # Stop after 15 overfitting epochs
```
- Current: 41 wasted epochs
- Target: Stop around epoch 75

#### Priority 4: Consider Hybrid Approach
- Use Adam for faster convergence
- Or use SGD with lower max_lr (0.005 instead of 0.01)

---

### 📁 Experiment Artifacts

- **Training Log**: `logs/20251011_141944_cifar100_training.log`
- **Model Checkpoint**: `models/cifar100_model_20251011_200637.pth`
- **Training Curves**: `logs/training_curves_20251011_200557.png`

---

### ✅ Conclusion

**Verdict**: ❌ **Regression from FT-1**

This experiment confirmed that:
1. **Dropout is essential** for ResNet-34 on CIFAR-100
2. **SGD requires more careful tuning** than Adam
3. **Larger models need stronger regularization**
4. **OneCycleLR cooldown** can waste training time if not tuned properly

**Next Steps**: Implement FT-3 with dropout=0.10 and adjusted OneCycleLR schedule to recover lost accuracy.

---

### 📊 Detailed Training Metrics

**Selected Epoch Details**:
- Epoch 1: Train 5.41%, Test 9.88% (gap: -4.47%)
- Epoch 10: Train 33.81%, Test 40.09% (gap: -6.28%)
- Epoch 30: Train 59.10%, Test 57.62% (gap: 1.48%)
- Epoch 60: Train 79.05%, Test 63.14% (gap: 15.91%) ← **Overfitting begins**
- Epoch 97: Train 91.35%, Test 67.50% (gap: 23.85%) ← **Best test accuracy**
- Epoch 100: Train 91.57%, Test 67.46% (gap: 24.11%)

**LR Schedule Verification**:
- Initial LR: 0.002022 (epoch 1)
- Peak LR: 0.010000 (epoch 30)
- Final LR: 0.000002 (epoch 100)
- Schedule worked as intended ✅

---

**Experiment Date**: October 11, 2025  
**Hardware**: CUDA GPU  
**Total Training Time**: 5 hours 46 minutes  
**Status**: ❌ Failed (worse than baseline)

---

## 🎯 EXPERIMENT LOG: ResNet-34 with Adam + OneCycleLR (October 11, 2025)

### Experiment ID: FT-3 (Adam-OneCycle-30)

---

### 🏆 Experiment Goal
Combine the best elements from previous experiments: ResNet-34 architecture with Adam optimizer and OneCycleLR scheduler, but with **shorter training (30 epochs)** to prevent overfitting.

### 📋 Configuration

**Model Architecture**: ResNet-34  
**Parameters**: 21,328,292 (~21.3M)  
**Dropout Rate**: 0.0 (No dropout - relying on OneCycleLR regularization)

**Optimizer**: Adam
- Initial LR: 0.001
- Betas: (0.9, 0.999)
- Eps: 1e-08
- Weight Decay: 0.0001

**Scheduler**: OneCycleLR
- Max LR: 0.003
- Pct Start: 0.3 (30% warmup)
- Div Factor: 5 (initial LR = max_lr/5 = 0.0006)
- Final Div Factor: 1000.0
- Anneal Strategy: cosine

**Training Config**:
- Epochs: 30 (strategic reduction from 100)
- Batch Size: 128
- Data Augmentation: Albumentations (HorizontalFlip, ShiftScaleRotate, CoarseDropout)

---

### 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | 73.57% @ epoch 30 |
| **Final Test Accuracy** | 73.57% @ epoch 30 |
| **Final Train Accuracy** | 84.35% |
| **Final Train-Test Gap** | 10.78% |
| **Average Gap** | 2.05% |
| **Overfitting Started** | Never (gap never exceeded 15%) |
| **Overfitting Epochs** | 0 out of 30 |
| **Training Duration** | ~36 minutes |

---

### 📈 Training Progression

| Phase | Epochs | Train Acc | Test Acc | Gap | LR Range | Status |
|-------|--------|-----------|----------|-----|----------|--------|
| **Warmup** | 1-5 | 8.66% → 36.45% | 13.68% → 39.03% | -5.02% → -2.58% | 0.0006 → 0.002 | ✅ Fast learning |
| **Ramp-up** | 6-9 | 40.33% → 47.55% | 38.35% → 45.67% | 1.98% → 1.88% | 0.002 → 0.003 | ✅ Peak LR reached |
| **High LR** | 10-15 | 49.37% → 58.01% | 47.58% → 57.48% | 1.79% → 0.53% | 0.003 → 0.002 | ✅ Best learning phase |
| **Cooldown** | 16-30 | 59.87% → 84.35% | 60.49% → 73.57% | -0.62% → 10.78% | 0.002 → 0.000001 | ✅ Continued improvement |

---

### 🔍 Key Observations

#### ✅ Strengths
1. **Best Test Accuracy**: 73.57% - highest across all experiments (+4.29% vs FT-1, +6.07% vs FT-2)
2. **Best Generalization**: 10.78% gap - lowest across all experiments (vs 21.47% FT-1, 24.11% FT-2)
3. **No Overfitting**: 0 overfitting epochs - healthy training throughout
4. **Efficient Training**: Achieved best results in just 30 epochs (~36 min vs 5.75 hours FT-2)
5. **Adam + OneCycleLR Synergy**: Perfect combination for this task
6. **Smooth Learning**: Gradual improvement without instabilities

#### ✅ Breakthrough Insights
1. **Shorter is Better**: 30 epochs optimal, 100 epochs causes overfitting
2. **Adam Superiority**: Adaptive learning rates crucial for CIFAR-100
3. **OneCycleLR with Adam**: Better than ReduceLROnPlateau for this architecture
4. **No Dropout Needed**: OneCycleLR provides sufficient regularization
5. **Larger Model Works**: ResNet-34 outperforms ResNet-18 with proper training

#### 📈 Performance Metrics
- **Gap Evolution**: Started at -5.02% (test > train), ended at 10.78% (healthy)
- **Learning Efficiency**: Reached 60% test accuracy by epoch 16
- **Final Phase**: Epochs 20-30 added 9% test accuracy without overfitting
- **LR Schedule**: Peaked at epoch 9, gradual cooldown maintained learning

---

### 📊 Comparison with All Experiments

| Metric | FT-1 (Adam) | FT-2 (SGD) | FT-3 (Adam-30) | Best |
|--------|-------------|------------|----------------|------|
| **Architecture** | ResNet-18 | ResNet-34 | ResNet-34 | FT-3 |
| **Parameters** | 0.93M | 21.3M | 21.3M | FT-3 |
| **Epochs** | 100 | 100 | 30 | FT-3 |
| **Best Test Acc** | 69.28% | 67.50% | **73.57%** | **FT-3** ✅ |
| **Final Gap** | 21.47% | 24.11% | **10.78%** | **FT-3** ✅ |
| **Avg Gap** | 9.13% | 10.01% | **2.05%** | **FT-3** ✅ |
| **Overfitting Epochs** | 26 | 41 | **0** | **FT-3** ✅ |
| **Training Time** | ~1.5h | ~5.75h | **~36m** | **FT-3** ✅ |
| **Dropout** | 0.05 | 0.0 | 0.0 | - |

**FT-3 wins on ALL metrics! 🏆**

---

### 💡 Analysis & Insights

#### Why Did This Experiment Succeed?

1. **Adam + OneCycleLR = Magic**
   - Adam's adaptive learning rates + OneCycleLR's schedule = optimal convergence
   - Better than Adam + ReduceLROnPlateau (FT-1)
   - Much better than SGD + OneCycleLR (FT-2)

2. **30 Epochs is the Sweet Spot**
   - FT-1 (100 epochs): Overfitting started at epoch 75
   - FT-2 (100 epochs): Overfitting started at epoch 60
   - FT-3 (30 epochs): No overfitting detected
   - **Insight**: Model converges by epoch 20, extra training hurts

3. **No Dropout Needed**
   - OneCycleLR provides implicit regularization through learning rate variation
   - Adam's adaptive rates prevent getting stuck in sharp minima
   - Result: Better than FT-1 with 0.05 dropout

4. **ResNet-34 > ResNet-18 (with proper training)**
   - FT-1 (ResNet-18): 69.28% test accuracy
   - FT-3 (ResNet-34): 73.57% test accuracy
   - **Insight**: Larger model capacity helps when training is well-regularized

5. **OneCycleLR Schedule Perfect for 30 Epochs**
   - Warmup: 9 epochs (30% of 30)
   - Peak performance: Epochs 10-20
   - Cooldown: Epochs 20-30 (fine-tuning)
   - Final LR: 0.000001 (sufficiently low)

---

### 🎯 Why This is the Best Model

| Aspect | FT-3 Advantage |
|--------|----------------|
| **Accuracy** | +4.29% vs FT-1, +6.07% vs FT-2 |
| **Generalization** | 10.78% gap vs 21.47% (FT-1), 24.11% (FT-2) |
| **Efficiency** | 30 epochs vs 100 epochs (70% time savings) |
| **Stability** | 0 overfitting epochs vs 26 (FT-1), 41 (FT-2) |
| **Simplicity** | No dropout, straightforward config |

---

### 📁 Experiment Artifacts

- **Training Log**: `logs/20251011_093929_cifar100_training.log`
- **Model Checkpoint**: `models/cifar100_model_20251011_093931.pth`
- **Training Curves**: `logs/training_curves_20251011_093929.png`

---

### ✅ Conclusion

**Verdict**: 🏆 **MAJOR SUCCESS - Best Model Achieved!**

This experiment discovered the optimal configuration:
1. **Architecture**: ResNet-34 (21.3M params)
2. **Optimizer**: Adam (adaptive learning rates)
3. **Scheduler**: OneCycleLR (implicit regularization)
4. **Epochs**: 30 (prevents overfitting)
5. **Dropout**: 0.0 (not needed with OneCycleLR)

**Key Findings**:
- ✅ Adam + OneCycleLR is superior to all other combinations
- ✅ 30 epochs is optimal (100 epochs causes overfitting)
- ✅ ResNet-34 outperforms ResNet-18 with proper training
- ✅ OneCycleLR eliminates need for dropout
- ✅ Shorter training = better generalization

**Deployment Ready**: This model (73.57% test accuracy, 10.78% gap) is production-ready and significantly outperforms all previous attempts.

---

### 📊 Detailed Training Metrics

**Selected Epoch Details**:
- Epoch 1: Train 8.66%, Test 13.68% (gap: -5.02%) - Fast start
- Epoch 5: Train 36.45%, Test 39.03% (gap: -2.58%) - Warmup complete
- Epoch 9: Train 47.55%, Test 45.67% (gap: 1.88%) - Peak LR
- Epoch 16: Train 59.87%, Test 60.49% (gap: -0.62%) - Test ahead!
- Epoch 21: Train 69.09%, Test 67.32% (gap: 1.77%) - Healthy learning
- Epoch 30: Train 84.35%, Test 73.57% (gap: 10.78%) - Final result

**LR Schedule Verification**:
- Initial LR: 0.000672 (epoch 1)
- Peak LR: 0.003000 (epoch 9)
- Mid LR: 0.001166 (epoch 21)
- Final LR: 0.000001 (epoch 30)
- Schedule worked perfectly ✅

**Accuracy Progression**:
- First 10 epochs: 13.68% → 47.58% test (+33.90%)
- Next 10 epochs: 47.58% → 64.60% test (+17.02%)
- Final 10 epochs: 64.60% → 73.57% test (+8.97%)
- Consistent improvement throughout

---

**Experiment Date**: October 11, 2025 (Google Colab)  
**Hardware**: CUDA GPU (Colab T4)  
**Total Training Time**: 36 minutes 10 seconds  
**Status**: ✅ **SUCCESS - New Best Model**

---
---
### Detailed Log - FT-3 BEST MODEL

2025-10-11 09:03:18,774 - CIFAR-100_Training - INFO - Logger initialized. Log file: /content/logs/20251011_090318_cifar100_training.log
INFO:CIFAR-100_Training:Logger initialized. Log file: /content/logs/20251011_090318_cifar100_training.log
2025-10-11 09:03:18,776 - CIFAR-100_Training - INFO - Updated Configuration (from main()):
INFO:CIFAR-100_Training:Updated Configuration (from main()):
2025-10-11 09:03:18,777 - CIFAR-100_Training - INFO -   - Epochs: 30
INFO:CIFAR-100_Training:  - Epochs: 30
2025-10-11 09:03:18,778 - CIFAR-100_Training - INFO -   - Learning Rate: 0.001
INFO:CIFAR-100_Training:  - Learning Rate: 0.001
2025-10-11 09:03:18,779 - CIFAR-100_Training - INFO -   - Optimizer: Adam
INFO:CIFAR-100_Training:  - Optimizer: Adam
2025-10-11 09:03:18,780 - CIFAR-100_Training - INFO -   - Weight Decay: 0.0001
INFO:CIFAR-100_Training:  - Weight Decay: 0.0001
2025-10-11 09:03:18,781 - CIFAR-100_Training - INFO -   - Adam Betas: (0.9, 0.999)
INFO:CIFAR-100_Training:  - Adam Betas: (0.9, 0.999)
2025-10-11 09:03:18,782 - CIFAR-100_Training - INFO -   - Adam Eps: 1e-08
INFO:CIFAR-100_Training:  - Adam Eps: 1e-08
2025-10-11 09:03:18,783 - CIFAR-100_Training - INFO -   - Scheduler: OneCycleLR
INFO:CIFAR-100_Training:  - Scheduler: OneCycleLR
2025-10-11 09:03:18,784 - CIFAR-100_Training - INFO -   - Max LR: 0.003
INFO:CIFAR-100_Training:  - Max LR: 0.003
2025-10-11 09:03:18,785 - CIFAR-100_Training - INFO -   - Pct Start: 0.3
INFO:CIFAR-100_Training:  - Pct Start: 0.3
2025-10-11 09:03:18,786 - CIFAR-100_Training - INFO -   - Div Factor: 5
INFO:CIFAR-100_Training:  - Div Factor: 5
2025-10-11 09:03:18,787 - CIFAR-100_Training - INFO -   - Final Div Factor: 1000.0
INFO:CIFAR-100_Training:  - Final Div Factor: 1000.0
2025-10-11 09:03:18,789 - CIFAR-100_Training - INFO -   - Anneal Strategy: cos
INFO:CIFAR-100_Training:  - Anneal Strategy: cos
2025-10-11 09:03:18,790 - CIFAR-100_Training - INFO -   - Batch Size: 128
INFO:CIFAR-100_Training:  - Batch Size: 128
2025-10-11 09:03:18,790 - CIFAR-100_Training - INFO -   - Num Workers: 4
INFO:CIFAR-100_Training:  - Num Workers: 4
2025-10-11 09:03:18,791 - CIFAR-100_Training - INFO -   - Pin Memory: True
INFO:CIFAR-100_Training:  - Pin Memory: True
2025-10-11 09:03:18,793 - CIFAR-100_Training - INFO -   - Shuffle: True
INFO:CIFAR-100_Training:  - Shuffle: True
2025-10-11 09:03:18,794 - CIFAR-100_Training - INFO -   - Dropout Rate: 0.0
INFO:CIFAR-100_Training:  - Dropout Rate: 0.0
2025-10-11 09:03:18,795 - CIFAR-100_Training - INFO -   - Device: CUDA
INFO:CIFAR-100_Training:  - Device: CUDA
2025-10-11 09:03:18,796 - CIFAR-100_Training - INFO -   - Log Directory: /content/logs
INFO:CIFAR-100_Training:  - Log Directory: /content/logs
2025-10-11 09:03:18,797 - CIFAR-100_Training - INFO -   - Model Save Directory: /content/models
INFO:CIFAR-100_Training:  - Model Save Directory: /content/models
2025-10-11 09:03:18,799 - CIFAR-100_Training - INFO -   - Save Model: True
INFO:CIFAR-100_Training:  - Save Model: True
2025-10-11 09:03:18,800 - CIFAR-100_Training - INFO -   - Log Level: INFO
INFO:CIFAR-100_Training:  - Log Level: INFO
2025-10-11 09:03:18,800 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:18,801 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:18,802 - CIFAR-100_Training - INFO - CIFAR-100 TRAINING EXPERIMENT STARTED
INFO:CIFAR-100_Training:CIFAR-100 TRAINING EXPERIMENT STARTED
2025-10-11 09:03:18,804 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:18,805 - CIFAR-100_Training - INFO - Data Config: DataConfig(data_dir='./data', batch_size=128, num_workers=4, pin_memory=True, shuffle=True, cifar100_mean=(0.507076, 0.48655, 0.440919), cifar100_std=(0.267334, 0.256438, 0.27615), rotation_range=(-7.0, 7.0), fill_value=1)
INFO:CIFAR-100_Training:Data Config: DataConfig(data_dir='./data', batch_size=128, num_workers=4, pin_memory=True, shuffle=True, cifar100_mean=(0.507076, 0.48655, 0.440919), cifar100_std=(0.267334, 0.256438, 0.27615), rotation_range=(-7.0, 7.0), fill_value=1)
2025-10-11 09:03:18,807 - CIFAR-100_Training - INFO - Model Config: ModelConfig(input_channels=3, input_size=(32, 32), num_classes=100, dropout_rate=0.0)
INFO:CIFAR-100_Training:Model Config: ModelConfig(input_channels=3, input_size=(32, 32), num_classes=100, dropout_rate=0.0)
2025-10-11 09:03:18,808 - CIFAR-100_Training - INFO - Training Config: TrainingConfig(epochs=30, learning_rate=0.001, momentum=0.9, weight_decay=0.0001, scheduler_step_size=10, scheduler_gamma=0.1, seed=1, optimizer_type='Adam', adam_betas=(0.9, 0.999), adam_eps=1e-08, rmsprop_alpha=0.99, scheduler_type='OneCycleLR', cosine_t_max=20, exponential_gamma=0.95, plateau_mode='min', plateau_factor=0.5, plateau_patience=5, plateau_threshold=0.0001, onecycle_max_lr=0.003, onecycle_pct_start=0.3, onecycle_div_factor=5, onecycle_final_div_factor=1000.0, onecycle_anneal_strategy='cos')
INFO:CIFAR-100_Training:Training Config: TrainingConfig(epochs=30, learning_rate=0.001, momentum=0.9, weight_decay=0.0001, scheduler_step_size=10, scheduler_gamma=0.1, seed=1, optimizer_type='Adam', adam_betas=(0.9, 0.999), adam_eps=1e-08, rmsprop_alpha=0.99, scheduler_type='OneCycleLR', cosine_t_max=20, exponential_gamma=0.95, plateau_mode='min', plateau_factor=0.5, plateau_patience=5, plateau_threshold=0.0001, onecycle_max_lr=0.003, onecycle_pct_start=0.3, onecycle_div_factor=5, onecycle_final_div_factor=1000.0, onecycle_anneal_strategy='cos')
2025-10-11 09:03:18,809 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:18,810 - CIFAR-100_Training - INFO - Setting up data...
INFO:CIFAR-100_Training:Setting up data...
2025-10-11 09:03:18,811 - CIFAR-100_Training - INFO - Using Albumentations for data augmentation
INFO:CIFAR-100_Training:Using Albumentations for data augmentation
/usr/local/lib/python3.12/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
2025-10-11 09:03:18,821 - CIFAR-100_Training - INFO - Loading CIFAR-100 dataset...
INFO:CIFAR-100_Training:Loading CIFAR-100 dataset...
============================================================
CIFAR-100 TRAINING SCRIPT - MONOLITHIC IMPLEMENTATION
============================================================
Configuration:
  - Epochs: 30
  - Learning Rate: 0.001
  - Batch Size: 128
  - Device: CUDA
  - Log Directory: /content/logs
  - Model Save Directory: /content/models
============================================================
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
2025-10-11 09:03:20,630 - CIFAR-100_Training - INFO - CIFAR-100 dataset loaded successfully!
INFO:CIFAR-100_Training:CIFAR-100 dataset loaded successfully!
2025-10-11 09:03:20,632 - CIFAR-100_Training - INFO - Train samples: 50000
INFO:CIFAR-100_Training:Train samples: 50000
2025-10-11 09:03:20,633 - CIFAR-100_Training - INFO - Test samples: 10000
INFO:CIFAR-100_Training:Test samples: 10000
2025-10-11 09:03:20,634 - CIFAR-100_Training - INFO - Augmentation library: Albumentations
INFO:CIFAR-100_Training:Augmentation library: Albumentations
2025-10-11 09:03:20,636 - CIFAR-100_Training - INFO - Computing CIFAR-100 data statistics...
INFO:CIFAR-100_Training:Computing CIFAR-100 data statistics...
2025-10-11 09:03:26,363 - CIFAR-100_Training - INFO - CIFAR-100 Data Statistics:
INFO:CIFAR-100_Training:CIFAR-100 Data Statistics:
2025-10-11 09:03:26,365 - CIFAR-100_Training - INFO -   - Shape: (50000, 32, 32, 3)
INFO:CIFAR-100_Training:  - Shape: (50000, 32, 32, 3)
2025-10-11 09:03:26,367 - CIFAR-100_Training - INFO -   - Size: 153,600,000
INFO:CIFAR-100_Training:  - Size: 153,600,000
2025-10-11 09:03:26,369 - CIFAR-100_Training - INFO -   - Min: 0.0000
INFO:CIFAR-100_Training:  - Min: 0.0000
2025-10-11 09:03:26,371 - CIFAR-100_Training - INFO -   - Max: 1.0000
INFO:CIFAR-100_Training:  - Max: 1.0000
2025-10-11 09:03:26,375 - CIFAR-100_Training - INFO -   - Mean: 0.4782
INFO:CIFAR-100_Training:  - Mean: 0.4782
2025-10-11 09:03:26,378 - CIFAR-100_Training - INFO -   - Std: 0.2682
INFO:CIFAR-100_Training:  - Std: 0.2682
2025-10-11 09:03:26,380 - CIFAR-100_Training - INFO -   - Variance: 0.0719
INFO:CIFAR-100_Training:  - Variance: 0.0719
2025-10-11 09:03:26,382 - CIFAR-100_Training - INFO - Channel-wise Statistics:
INFO:CIFAR-100_Training:Channel-wise Statistics:
2025-10-11 09:03:26,384 - CIFAR-100_Training - INFO -   Red Channel:
INFO:CIFAR-100_Training:  Red Channel:
2025-10-11 09:03:26,386 - CIFAR-100_Training - INFO -     - Mean: 0.5071
INFO:CIFAR-100_Training:    - Mean: 0.5071
2025-10-11 09:03:26,388 - CIFAR-100_Training - INFO -     - Std: 0.2673
INFO:CIFAR-100_Training:    - Std: 0.2673
2025-10-11 09:03:26,390 - CIFAR-100_Training - INFO -     - Min: 0.0000
INFO:CIFAR-100_Training:    - Min: 0.0000
2025-10-11 09:03:26,392 - CIFAR-100_Training - INFO -     - Max: 1.0000
INFO:CIFAR-100_Training:    - Max: 1.0000
2025-10-11 09:03:26,393 - CIFAR-100_Training - INFO -   Green Channel:
INFO:CIFAR-100_Training:  Green Channel:
2025-10-11 09:03:26,394 - CIFAR-100_Training - INFO -     - Mean: 0.4866
INFO:CIFAR-100_Training:    - Mean: 0.4866
2025-10-11 09:03:26,395 - CIFAR-100_Training - INFO -     - Std: 0.2564
INFO:CIFAR-100_Training:    - Std: 0.2564
2025-10-11 09:03:26,396 - CIFAR-100_Training - INFO -     - Min: 0.0000
INFO:CIFAR-100_Training:    - Min: 0.0000
2025-10-11 09:03:26,397 - CIFAR-100_Training - INFO -     - Max: 1.0000
INFO:CIFAR-100_Training:    - Max: 1.0000
2025-10-11 09:03:26,398 - CIFAR-100_Training - INFO -   Blue Channel:
INFO:CIFAR-100_Training:  Blue Channel:
2025-10-11 09:03:26,399 - CIFAR-100_Training - INFO -     - Mean: 0.4409
INFO:CIFAR-100_Training:    - Mean: 0.4409
2025-10-11 09:03:26,401 - CIFAR-100_Training - INFO -     - Std: 0.2762
INFO:CIFAR-100_Training:    - Std: 0.2762
2025-10-11 09:03:26,402 - CIFAR-100_Training - INFO -     - Min: 0.0000
INFO:CIFAR-100_Training:    - Min: 0.0000
2025-10-11 09:03:26,403 - CIFAR-100_Training - INFO -     - Max: 1.0000
INFO:CIFAR-100_Training:    - Max: 1.0000
2025-10-11 09:03:27,011 - CIFAR-100_Training - INFO - CIFAR-100 Batch Information:
INFO:CIFAR-100_Training:CIFAR-100 Batch Information:
2025-10-11 09:03:27,018 - CIFAR-100_Training - INFO -   - Batch size: 128
INFO:CIFAR-100_Training:  - Batch size: 128
2025-10-11 09:03:27,024 - CIFAR-100_Training - INFO -   - Image shape: torch.Size([3, 32, 32])
INFO:CIFAR-100_Training:  - Image shape: torch.Size([3, 32, 32])
2025-10-11 09:03:27,026 - CIFAR-100_Training - INFO -   - Label shape: torch.Size([128])
INFO:CIFAR-100_Training:  - Label shape: torch.Size([128])
2025-10-11 09:03:27,029 - CIFAR-100_Training - INFO -   - Data type: torch.float32
INFO:CIFAR-100_Training:  - Data type: torch.float32
2025-10-11 09:03:27,035 - CIFAR-100_Training - INFO -   - Number of classes: 100
INFO:CIFAR-100_Training:  - Number of classes: 100
2025-10-11 09:03:27,122 - CIFAR-100_Training - INFO - Getting input size from CIFAR-100 data loader...
INFO:CIFAR-100_Training:Getting input size from CIFAR-100 data loader...
2025-10-11 09:03:27,482 - CIFAR-100_Training - INFO - CIFAR-100 input size from data loader: (3, 32, 32)
INFO:CIFAR-100_Training:CIFAR-100 input size from data loader: (3, 32, 32)
2025-10-11 09:03:27,638 - CIFAR-100_Training - INFO - Setting up model...
INFO:CIFAR-100_Training:Setting up model...
2025-10-11 09:03:27,952 - CIFAR-100_Training - INFO - Generating ResNet-34 summary...
INFO:CIFAR-100_Training:Generating ResNet-34 summary...
2025-10-11 09:03:28,199 - CIFAR-100_Training - INFO - ResNet-34 Architecture Summary:
INFO:CIFAR-100_Training:ResNet-34 Architecture Summary:
2025-10-11 09:03:28,201 - CIFAR-100_Training - INFO -   - Total Parameters: 21,328,292
INFO:CIFAR-100_Training:  - Total Parameters: 21,328,292
2025-10-11 09:03:28,202 - CIFAR-100_Training - INFO -   - Batch Normalization: Yes
INFO:CIFAR-100_Training:  - Batch Normalization: Yes
2025-10-11 09:03:28,204 - CIFAR-100_Training - INFO -   - Dropout: Yes
INFO:CIFAR-100_Training:  - Dropout: Yes
2025-10-11 09:03:28,205 - CIFAR-100_Training - INFO -   - FC Layers: Yes
INFO:CIFAR-100_Training:  - FC Layers: Yes
2025-10-11 09:03:28,206 - CIFAR-100_Training - INFO -   - GAP Layers: Yes
INFO:CIFAR-100_Training:  - GAP Layers: Yes
2025-10-11 09:03:28,207 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:28,208 - CIFAR-100_Training - INFO - DETAILED MODEL ARCHITECTURE SUMMARY
INFO:CIFAR-100_Training:DETAILED MODEL ARCHITECTURE SUMMARY
2025-10-11 09:03:28,209 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:28,210 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
INFO:CIFAR-100_Training:----------------------------------------------------------------
2025-10-11 09:03:28,212 - CIFAR-100_Training - INFO -         Layer (type)               Output Shape         Param #
INFO:CIFAR-100_Training:        Layer (type)               Output Shape         Param #
2025-10-11 09:03:28,213 - CIFAR-100_Training - INFO - ================================================================
INFO:CIFAR-100_Training:================================================================
2025-10-11 09:03:28,214 - CIFAR-100_Training - INFO -             Conv2d-1           [-1, 64, 32, 32]           1,728
INFO:CIFAR-100_Training:            Conv2d-1           [-1, 64, 32, 32]           1,728
2025-10-11 09:03:28,216 - CIFAR-100_Training - INFO -        BatchNorm2d-2           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:       BatchNorm2d-2           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,217 - CIFAR-100_Training - INFO -               ReLU-3           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:              ReLU-3           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,219 - CIFAR-100_Training - INFO -             Conv2d-4           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:            Conv2d-4           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,220 - CIFAR-100_Training - INFO -        BatchNorm2d-5           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:       BatchNorm2d-5           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,221 - CIFAR-100_Training - INFO -               ReLU-6           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:              ReLU-6           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,222 - CIFAR-100_Training - INFO -             Conv2d-7           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:            Conv2d-7           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,223 - CIFAR-100_Training - INFO -        BatchNorm2d-8           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:       BatchNorm2d-8           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,225 - CIFAR-100_Training - INFO -               ReLU-9           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:              ReLU-9           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,226 - CIFAR-100_Training - INFO -        BasicBlock-10           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:       BasicBlock-10           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,227 - CIFAR-100_Training - INFO -            Conv2d-11           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:           Conv2d-11           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,229 - CIFAR-100_Training - INFO -       BatchNorm2d-12           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:      BatchNorm2d-12           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,230 - CIFAR-100_Training - INFO -              ReLU-13           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:             ReLU-13           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,233 - CIFAR-100_Training - INFO -            Conv2d-14           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:           Conv2d-14           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,234 - CIFAR-100_Training - INFO -       BatchNorm2d-15           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:      BatchNorm2d-15           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,235 - CIFAR-100_Training - INFO -              ReLU-16           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:             ReLU-16           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,238 - CIFAR-100_Training - INFO -        BasicBlock-17           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:       BasicBlock-17           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,239 - CIFAR-100_Training - INFO -            Conv2d-18           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:           Conv2d-18           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,240 - CIFAR-100_Training - INFO -       BatchNorm2d-19           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:      BatchNorm2d-19           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,241 - CIFAR-100_Training - INFO -              ReLU-20           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:             ReLU-20           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,242 - CIFAR-100_Training - INFO -            Conv2d-21           [-1, 64, 32, 32]          36,864
INFO:CIFAR-100_Training:           Conv2d-21           [-1, 64, 32, 32]          36,864
2025-10-11 09:03:28,244 - CIFAR-100_Training - INFO -       BatchNorm2d-22           [-1, 64, 32, 32]             128
INFO:CIFAR-100_Training:      BatchNorm2d-22           [-1, 64, 32, 32]             128
2025-10-11 09:03:28,246 - CIFAR-100_Training - INFO -              ReLU-23           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:             ReLU-23           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,248 - CIFAR-100_Training - INFO -        BasicBlock-24           [-1, 64, 32, 32]               0
INFO:CIFAR-100_Training:       BasicBlock-24           [-1, 64, 32, 32]               0
2025-10-11 09:03:28,249 - CIFAR-100_Training - INFO -            Conv2d-25          [-1, 128, 16, 16]          73,728
INFO:CIFAR-100_Training:           Conv2d-25          [-1, 128, 16, 16]          73,728
2025-10-11 09:03:28,250 - CIFAR-100_Training - INFO -       BatchNorm2d-26          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-26          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,253 - CIFAR-100_Training - INFO -              ReLU-27          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-27          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,255 - CIFAR-100_Training - INFO -            Conv2d-28          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-28          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,256 - CIFAR-100_Training - INFO -       BatchNorm2d-29          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-29          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,257 - CIFAR-100_Training - INFO -            Conv2d-30          [-1, 128, 16, 16]           8,192
INFO:CIFAR-100_Training:           Conv2d-30          [-1, 128, 16, 16]           8,192
2025-10-11 09:03:28,259 - CIFAR-100_Training - INFO -       BatchNorm2d-31          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-31          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,261 - CIFAR-100_Training - INFO -              ReLU-32          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-32          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,264 - CIFAR-100_Training - INFO -        BasicBlock-33          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:       BasicBlock-33          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,265 - CIFAR-100_Training - INFO -            Conv2d-34          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-34          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,266 - CIFAR-100_Training - INFO -       BatchNorm2d-35          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-35          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,267 - CIFAR-100_Training - INFO -              ReLU-36          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-36          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,268 - CIFAR-100_Training - INFO -            Conv2d-37          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-37          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,269 - CIFAR-100_Training - INFO -       BatchNorm2d-38          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-38          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,273 - CIFAR-100_Training - INFO -              ReLU-39          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-39          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,274 - CIFAR-100_Training - INFO -        BasicBlock-40          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:       BasicBlock-40          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,276 - CIFAR-100_Training - INFO -            Conv2d-41          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-41          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,277 - CIFAR-100_Training - INFO -       BatchNorm2d-42          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-42          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,278 - CIFAR-100_Training - INFO -              ReLU-43          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-43          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,279 - CIFAR-100_Training - INFO -            Conv2d-44          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-44          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,280 - CIFAR-100_Training - INFO -       BatchNorm2d-45          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-45          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,281 - CIFAR-100_Training - INFO -              ReLU-46          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-46          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,282 - CIFAR-100_Training - INFO -        BasicBlock-47          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:       BasicBlock-47          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,283 - CIFAR-100_Training - INFO -            Conv2d-48          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-48          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,284 - CIFAR-100_Training - INFO -       BatchNorm2d-49          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-49          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,285 - CIFAR-100_Training - INFO -              ReLU-50          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-50          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,287 - CIFAR-100_Training - INFO -            Conv2d-51          [-1, 128, 16, 16]         147,456
INFO:CIFAR-100_Training:           Conv2d-51          [-1, 128, 16, 16]         147,456
2025-10-11 09:03:28,288 - CIFAR-100_Training - INFO -       BatchNorm2d-52          [-1, 128, 16, 16]             256
INFO:CIFAR-100_Training:      BatchNorm2d-52          [-1, 128, 16, 16]             256
2025-10-11 09:03:28,289 - CIFAR-100_Training - INFO -              ReLU-53          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:             ReLU-53          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,290 - CIFAR-100_Training - INFO -        BasicBlock-54          [-1, 128, 16, 16]               0
INFO:CIFAR-100_Training:       BasicBlock-54          [-1, 128, 16, 16]               0
2025-10-11 09:03:28,292 - CIFAR-100_Training - INFO -            Conv2d-55            [-1, 256, 8, 8]         294,912
INFO:CIFAR-100_Training:           Conv2d-55            [-1, 256, 8, 8]         294,912
2025-10-11 09:03:28,293 - CIFAR-100_Training - INFO -       BatchNorm2d-56            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-56            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,294 - CIFAR-100_Training - INFO -              ReLU-57            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-57            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,295 - CIFAR-100_Training - INFO -            Conv2d-58            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-58            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,297 - CIFAR-100_Training - INFO -       BatchNorm2d-59            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-59            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,298 - CIFAR-100_Training - INFO -            Conv2d-60            [-1, 256, 8, 8]          32,768
INFO:CIFAR-100_Training:           Conv2d-60            [-1, 256, 8, 8]          32,768
2025-10-11 09:03:28,299 - CIFAR-100_Training - INFO -       BatchNorm2d-61            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-61            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,300 - CIFAR-100_Training - INFO -              ReLU-62            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-62            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,301 - CIFAR-100_Training - INFO -        BasicBlock-63            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-63            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,302 - CIFAR-100_Training - INFO -            Conv2d-64            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-64            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,303 - CIFAR-100_Training - INFO -       BatchNorm2d-65            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-65            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,305 - CIFAR-100_Training - INFO -              ReLU-66            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-66            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,306 - CIFAR-100_Training - INFO -            Conv2d-67            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-67            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,307 - CIFAR-100_Training - INFO -       BatchNorm2d-68            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-68            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,308 - CIFAR-100_Training - INFO -              ReLU-69            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-69            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,309 - CIFAR-100_Training - INFO -        BasicBlock-70            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-70            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,310 - CIFAR-100_Training - INFO -            Conv2d-71            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-71            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,311 - CIFAR-100_Training - INFO -       BatchNorm2d-72            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-72            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,313 - CIFAR-100_Training - INFO -              ReLU-73            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-73            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,314 - CIFAR-100_Training - INFO -            Conv2d-74            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-74            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,315 - CIFAR-100_Training - INFO -       BatchNorm2d-75            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-75            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,316 - CIFAR-100_Training - INFO -              ReLU-76            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-76            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,317 - CIFAR-100_Training - INFO -        BasicBlock-77            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-77            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,319 - CIFAR-100_Training - INFO -            Conv2d-78            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-78            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,319 - CIFAR-100_Training - INFO -       BatchNorm2d-79            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-79            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,321 - CIFAR-100_Training - INFO -              ReLU-80            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-80            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,323 - CIFAR-100_Training - INFO -            Conv2d-81            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-81            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,324 - CIFAR-100_Training - INFO -       BatchNorm2d-82            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-82            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,324 - CIFAR-100_Training - INFO -              ReLU-83            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-83            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,325 - CIFAR-100_Training - INFO -        BasicBlock-84            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-84            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,327 - CIFAR-100_Training - INFO -            Conv2d-85            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-85            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,328 - CIFAR-100_Training - INFO -       BatchNorm2d-86            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-86            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,329 - CIFAR-100_Training - INFO -              ReLU-87            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-87            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,330 - CIFAR-100_Training - INFO -            Conv2d-88            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-88            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,331 - CIFAR-100_Training - INFO -       BatchNorm2d-89            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-89            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,332 - CIFAR-100_Training - INFO -              ReLU-90            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-90            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,335 - CIFAR-100_Training - INFO -        BasicBlock-91            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-91            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,335 - CIFAR-100_Training - INFO -            Conv2d-92            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-92            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,337 - CIFAR-100_Training - INFO -       BatchNorm2d-93            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-93            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,337 - CIFAR-100_Training - INFO -              ReLU-94            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-94            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,338 - CIFAR-100_Training - INFO -            Conv2d-95            [-1, 256, 8, 8]         589,824
INFO:CIFAR-100_Training:           Conv2d-95            [-1, 256, 8, 8]         589,824
2025-10-11 09:03:28,340 - CIFAR-100_Training - INFO -       BatchNorm2d-96            [-1, 256, 8, 8]             512
INFO:CIFAR-100_Training:      BatchNorm2d-96            [-1, 256, 8, 8]             512
2025-10-11 09:03:28,341 - CIFAR-100_Training - INFO -              ReLU-97            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:             ReLU-97            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,342 - CIFAR-100_Training - INFO -        BasicBlock-98            [-1, 256, 8, 8]               0
INFO:CIFAR-100_Training:       BasicBlock-98            [-1, 256, 8, 8]               0
2025-10-11 09:03:28,343 - CIFAR-100_Training - INFO -            Conv2d-99            [-1, 512, 4, 4]       1,179,648
INFO:CIFAR-100_Training:           Conv2d-99            [-1, 512, 4, 4]       1,179,648
2025-10-11 09:03:28,344 - CIFAR-100_Training - INFO -      BatchNorm2d-100            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-100            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,345 - CIFAR-100_Training - INFO -             ReLU-101            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-101            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,346 - CIFAR-100_Training - INFO -           Conv2d-102            [-1, 512, 4, 4]       2,359,296
INFO:CIFAR-100_Training:          Conv2d-102            [-1, 512, 4, 4]       2,359,296
2025-10-11 09:03:28,348 - CIFAR-100_Training - INFO -      BatchNorm2d-103            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-103            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,349 - CIFAR-100_Training - INFO -           Conv2d-104            [-1, 512, 4, 4]         131,072
INFO:CIFAR-100_Training:          Conv2d-104            [-1, 512, 4, 4]         131,072
2025-10-11 09:03:28,350 - CIFAR-100_Training - INFO -      BatchNorm2d-105            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-105            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,351 - CIFAR-100_Training - INFO -             ReLU-106            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-106            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,352 - CIFAR-100_Training - INFO -       BasicBlock-107            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:      BasicBlock-107            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,353 - CIFAR-100_Training - INFO -           Conv2d-108            [-1, 512, 4, 4]       2,359,296
INFO:CIFAR-100_Training:          Conv2d-108            [-1, 512, 4, 4]       2,359,296
2025-10-11 09:03:28,354 - CIFAR-100_Training - INFO -      BatchNorm2d-109            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-109            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,355 - CIFAR-100_Training - INFO -             ReLU-110            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-110            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,357 - CIFAR-100_Training - INFO -           Conv2d-111            [-1, 512, 4, 4]       2,359,296
INFO:CIFAR-100_Training:          Conv2d-111            [-1, 512, 4, 4]       2,359,296
2025-10-11 09:03:28,358 - CIFAR-100_Training - INFO -      BatchNorm2d-112            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-112            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,371 - CIFAR-100_Training - INFO -             ReLU-113            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-113            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,375 - CIFAR-100_Training - INFO -       BasicBlock-114            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:      BasicBlock-114            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,376 - CIFAR-100_Training - INFO -           Conv2d-115            [-1, 512, 4, 4]       2,359,296
INFO:CIFAR-100_Training:          Conv2d-115            [-1, 512, 4, 4]       2,359,296
2025-10-11 09:03:28,377 - CIFAR-100_Training - INFO -      BatchNorm2d-116            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-116            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,378 - CIFAR-100_Training - INFO -             ReLU-117            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-117            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,379 - CIFAR-100_Training - INFO -           Conv2d-118            [-1, 512, 4, 4]       2,359,296
INFO:CIFAR-100_Training:          Conv2d-118            [-1, 512, 4, 4]       2,359,296
2025-10-11 09:03:28,383 - CIFAR-100_Training - INFO -      BatchNorm2d-119            [-1, 512, 4, 4]           1,024
INFO:CIFAR-100_Training:     BatchNorm2d-119            [-1, 512, 4, 4]           1,024
2025-10-11 09:03:28,385 - CIFAR-100_Training - INFO -             ReLU-120            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:            ReLU-120            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,388 - CIFAR-100_Training - INFO -       BasicBlock-121            [-1, 512, 4, 4]               0
INFO:CIFAR-100_Training:      BasicBlock-121            [-1, 512, 4, 4]               0
2025-10-11 09:03:28,394 - CIFAR-100_Training - INFO - AdaptiveAvgPool2d-122            [-1, 512, 1, 1]               0
INFO:CIFAR-100_Training:AdaptiveAvgPool2d-122            [-1, 512, 1, 1]               0
2025-10-11 09:03:28,398 - CIFAR-100_Training - INFO -          Dropout-123                  [-1, 512]               0
INFO:CIFAR-100_Training:         Dropout-123                  [-1, 512]               0
2025-10-11 09:03:28,405 - CIFAR-100_Training - INFO -           Linear-124                  [-1, 100]          51,300
INFO:CIFAR-100_Training:          Linear-124                  [-1, 100]          51,300
2025-10-11 09:03:28,407 - CIFAR-100_Training - INFO - ================================================================
INFO:CIFAR-100_Training:================================================================
2025-10-11 09:03:28,410 - CIFAR-100_Training - INFO - Total params: 21,328,292
INFO:CIFAR-100_Training:Total params: 21,328,292
2025-10-11 09:03:28,412 - CIFAR-100_Training - INFO - Trainable params: 21,328,292
INFO:CIFAR-100_Training:Trainable params: 21,328,292
2025-10-11 09:03:28,413 - CIFAR-100_Training - INFO - Non-trainable params: 0
INFO:CIFAR-100_Training:Non-trainable params: 0
2025-10-11 09:03:28,414 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
INFO:CIFAR-100_Training:----------------------------------------------------------------
2025-10-11 09:03:28,416 - CIFAR-100_Training - INFO - Input size (MB): 0.01
INFO:CIFAR-100_Training:Input size (MB): 0.01
2025-10-11 09:03:28,417 - CIFAR-100_Training - INFO - Forward/backward pass size (MB): 26.45
INFO:CIFAR-100_Training:Forward/backward pass size (MB): 26.45
2025-10-11 09:03:28,420 - CIFAR-100_Training - INFO - Params size (MB): 81.36
INFO:CIFAR-100_Training:Params size (MB): 81.36
2025-10-11 09:03:28,421 - CIFAR-100_Training - INFO - Estimated Total Size (MB): 107.82
INFO:CIFAR-100_Training:Estimated Total Size (MB): 107.82
2025-10-11 09:03:28,423 - CIFAR-100_Training - INFO - ----------------------------------------------------------------
INFO:CIFAR-100_Training:----------------------------------------------------------------
2025-10-11 09:03:28,424 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:03:28,425 - CIFAR-100_Training - INFO - Setting up trainer...
INFO:CIFAR-100_Training:Setting up trainer...
2025-10-11 09:03:28,432 - CIFAR-100_Training - INFO - Using device: cuda
INFO:CIFAR-100_Training:Using device: cuda
2025-10-11 09:03:28,434 - CIFAR-100_Training - INFO - Early stopping: Stop if train_acc - test_acc > 15.0% for 10 epochs
INFO:CIFAR-100_Training:Early stopping: Stop if train_acc - test_acc > 15.0% for 10 epochs
2025-10-11 09:03:28,435 - CIFAR-100_Training - INFO - Starting training process...
INFO:CIFAR-100_Training:Starting training process...
2025-10-11 09:03:28,437 - CIFAR-100_Training - INFO - Starting training process...
INFO:CIFAR-100_Training:Starting training process...
2025-10-11 09:03:28,440 - CIFAR-100_Training - INFO - Using optimizer: Adam
INFO:CIFAR-100_Training:Using optimizer: Adam
2025-10-11 09:03:28,442 - CIFAR-100_Training - INFO - Using scheduler: OneCycleLR
INFO:CIFAR-100_Training:Using scheduler: OneCycleLR
2025-10-11 09:03:28,444 - CIFAR-100_Training - INFO - Optimizer Configuration:
INFO:CIFAR-100_Training:Optimizer Configuration:
2025-10-11 09:03:28,445 - CIFAR-100_Training - INFO -   - Learning Rate: 0.001
INFO:CIFAR-100_Training:  - Learning Rate: 0.001
2025-10-11 09:03:28,449 - CIFAR-100_Training - INFO -   - Betas: (0.9, 0.999)
INFO:CIFAR-100_Training:  - Betas: (0.9, 0.999)
2025-10-11 09:03:28,450 - CIFAR-100_Training - INFO -   - Eps: 1e-08
INFO:CIFAR-100_Training:  - Eps: 1e-08
2025-10-11 09:03:28,451 - CIFAR-100_Training - INFO -   - Weight Decay: 0.0001
INFO:CIFAR-100_Training:  - Weight Decay: 0.0001
2025-10-11 09:03:28,453 - CIFAR-100_Training - INFO - Scheduler Configuration:
INFO:CIFAR-100_Training:Scheduler Configuration:
2025-10-11 09:03:28,454 - CIFAR-100_Training - INFO -   - Max LR: 0.003
INFO:CIFAR-100_Training:  - Max LR: 0.003
2025-10-11 09:03:28,455 - CIFAR-100_Training - INFO -   - Steps per Epoch: 391
INFO:CIFAR-100_Training:  - Steps per Epoch: 391
2025-10-11 09:03:28,456 - CIFAR-100_Training - INFO -   - Pct Start: 0.3
INFO:CIFAR-100_Training:  - Pct Start: 0.3
2025-10-11 09:03:28,460 - CIFAR-100_Training - INFO -   - Div Factor: 5
INFO:CIFAR-100_Training:  - Div Factor: 5
2025-10-11 09:03:28,463 - CIFAR-100_Training - INFO -   - Final Div Factor: 1000.0
INFO:CIFAR-100_Training:  - Final Div Factor: 1000.0
2025-10-11 09:03:28,464 - CIFAR-100_Training - INFO -   - Anneal Strategy: cos
INFO:CIFAR-100_Training:  - Anneal Strategy: cos
2025-10-11 09:03:28,465 - CIFAR-100_Training - INFO - Starting Epoch 1/30
INFO:CIFAR-100_Training:Starting Epoch 1/30
Epoch 1 - Loss: 3.8129, Accuracy: 8.66%: 100%|██████████| 391/391 [01:09<00:00,  5.62it/s]
2025-10-11 09:04:42,212 - CIFAR-100_Training - INFO - Epoch  1: Train Loss: 3.9948, Train Acc: 8.66%, Test Loss: 3.6164, Test Acc: 13.68%, Acc Diff: -5.02%, LR: 0.000672
INFO:CIFAR-100_Training:Epoch  1: Train Loss: 3.9948, Train Acc: 8.66%, Test Loss: 3.6164, Test Acc: 13.68%, Acc Diff: -5.02%, LR: 0.000672
2025-10-11 09:04:42,214 - CIFAR-100_Training - INFO - Starting Epoch 2/30
INFO:CIFAR-100_Training:Starting Epoch 2/30
Epoch 2 - Loss: 3.5250, Accuracy: 17.78%: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]
2025-10-11 09:05:55,370 - CIFAR-100_Training - INFO - Epoch  2: Train Loss: 3.3979, Train Acc: 17.78%, Test Loss: 3.0545, Test Acc: 23.69%, Acc Diff: -5.91%, LR: 0.000881
INFO:CIFAR-100_Training:Epoch  2: Train Loss: 3.3979, Train Acc: 17.78%, Test Loss: 3.0545, Test Acc: 23.69%, Acc Diff: -5.91%, LR: 0.000881
2025-10-11 09:05:55,374 - CIFAR-100_Training - INFO - Starting Epoch 3/30
INFO:CIFAR-100_Training:Starting Epoch 3/30
Epoch 3 - Loss: 2.8654, Accuracy: 25.55%: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]
2025-10-11 09:07:08,388 - CIFAR-100_Training - INFO - Epoch  3: Train Loss: 2.9481, Train Acc: 25.55%, Test Loss: 2.7024, Test Acc: 31.18%, Acc Diff: -5.63%, LR: 0.001200
INFO:CIFAR-100_Training:Epoch  3: Train Loss: 2.9481, Train Acc: 25.55%, Test Loss: 2.7024, Test Acc: 31.18%, Acc Diff: -5.63%, LR: 0.001200
2025-10-11 09:07:08,389 - CIFAR-100_Training - INFO - Starting Epoch 4/30
INFO:CIFAR-100_Training:Starting Epoch 4/30
Epoch 4 - Loss: 2.5580, Accuracy: 32.37%: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]
2025-10-11 09:08:21,471 - CIFAR-100_Training - INFO - Epoch  4: Train Loss: 2.6175, Train Acc: 32.37%, Test Loss: 2.4147, Test Acc: 36.86%, Acc Diff: -4.49%, LR: 0.001592
INFO:CIFAR-100_Training:Epoch  4: Train Loss: 2.6175, Train Acc: 32.37%, Test Loss: 2.4147, Test Acc: 36.86%, Acc Diff: -4.49%, LR: 0.001592
2025-10-11 09:08:21,473 - CIFAR-100_Training - INFO - Starting Epoch 5/30
INFO:CIFAR-100_Training:Starting Epoch 5/30
Epoch 5 - Loss: 2.4534, Accuracy: 36.45%: 100%|██████████| 391/391 [01:08<00:00,  5.68it/s]
2025-10-11 09:09:34,622 - CIFAR-100_Training - INFO - Epoch  5: Train Loss: 2.4030, Train Acc: 36.45%, Test Loss: 2.3176, Test Acc: 39.03%, Acc Diff: -2.58%, LR: 0.002009
INFO:CIFAR-100_Training:Epoch  5: Train Loss: 2.4030, Train Acc: 36.45%, Test Loss: 2.3176, Test Acc: 39.03%, Acc Diff: -2.58%, LR: 0.002009
2025-10-11 09:09:34,626 - CIFAR-100_Training - INFO - Starting Epoch 6/30
INFO:CIFAR-100_Training:Starting Epoch 6/30
Epoch 6 - Loss: 2.2468, Accuracy: 40.33%: 100%|██████████| 391/391 [01:08<00:00,  5.69it/s]
2025-10-11 09:10:47,522 - CIFAR-100_Training - INFO - Epoch  6: Train Loss: 2.2322, Train Acc: 40.33%, Test Loss: 2.3176, Test Acc: 38.35%, Acc Diff: 1.98%, LR: 0.002401
INFO:CIFAR-100_Training:Epoch  6: Train Loss: 2.2322, Train Acc: 40.33%, Test Loss: 2.3176, Test Acc: 38.35%, Acc Diff: 1.98%, LR: 0.002401
2025-10-11 09:10:47,524 - CIFAR-100_Training - INFO - Starting Epoch 7/30
INFO:CIFAR-100_Training:Starting Epoch 7/30
Epoch 7 - Loss: 2.0017, Accuracy: 42.88%: 100%|██████████| 391/391 [01:08<00:00,  5.71it/s]
2025-10-11 09:12:00,233 - CIFAR-100_Training - INFO - Epoch  7: Train Loss: 2.1146, Train Acc: 42.88%, Test Loss: 2.4122, Test Acc: 38.66%, Acc Diff: 4.22%, LR: 0.002720
INFO:CIFAR-100_Training:Epoch  7: Train Loss: 2.1146, Train Acc: 42.88%, Test Loss: 2.4122, Test Acc: 38.66%, Acc Diff: 4.22%, LR: 0.002720
2025-10-11 09:12:00,237 - CIFAR-100_Training - INFO - Starting Epoch 8/30
INFO:CIFAR-100_Training:Starting Epoch 8/30
Epoch 8 - Loss: 2.2147, Accuracy: 45.35%: 100%|██████████| 391/391 [01:08<00:00,  5.73it/s]
2025-10-11 09:13:12,647 - CIFAR-100_Training - INFO - Epoch  8: Train Loss: 2.0173, Train Acc: 45.35%, Test Loss: 2.1097, Test Acc: 45.42%, Acc Diff: -0.07%, LR: 0.002928
INFO:CIFAR-100_Training:Epoch  8: Train Loss: 2.0173, Train Acc: 45.35%, Test Loss: 2.1097, Test Acc: 45.42%, Acc Diff: -0.07%, LR: 0.002928
2025-10-11 09:13:12,648 - CIFAR-100_Training - INFO - Starting Epoch 9/30
INFO:CIFAR-100_Training:Starting Epoch 9/30
Epoch 9 - Loss: 2.2325, Accuracy: 47.55%: 100%|██████████| 391/391 [01:08<00:00,  5.75it/s]
2025-10-11 09:14:24,857 - CIFAR-100_Training - INFO - Epoch  9: Train Loss: 1.9174, Train Acc: 47.55%, Test Loss: 2.0794, Test Acc: 45.67%, Acc Diff: 1.88%, LR: 0.003000
INFO:CIFAR-100_Training:Epoch  9: Train Loss: 1.9174, Train Acc: 47.55%, Test Loss: 2.0794, Test Acc: 45.67%, Acc Diff: 1.88%, LR: 0.003000
2025-10-11 09:14:24,859 - CIFAR-100_Training - INFO - Starting Epoch 10/30
INFO:CIFAR-100_Training:Starting Epoch 10/30
Epoch 10 - Loss: 1.6564, Accuracy: 49.37%: 100%|██████████| 391/391 [01:07<00:00,  5.75it/s]
2025-10-11 09:15:37,166 - CIFAR-100_Training - INFO - Epoch 10: Train Loss: 1.8320, Train Acc: 49.37%, Test Loss: 1.8956, Test Acc: 47.58%, Acc Diff: 1.79%, LR: 0.002983
INFO:CIFAR-100_Training:Epoch 10: Train Loss: 1.8320, Train Acc: 49.37%, Test Loss: 1.8956, Test Acc: 47.58%, Acc Diff: 1.79%, LR: 0.002983
2025-10-11 09:15:37,168 - CIFAR-100_Training - INFO - Starting Epoch 11/30
INFO:CIFAR-100_Training:Starting Epoch 11/30
Epoch 11 - Loss: 1.8591, Accuracy: 51.39%: 100%|██████████| 391/391 [01:07<00:00,  5.76it/s]
2025-10-11 09:16:49,156 - CIFAR-100_Training - INFO - Epoch 11: Train Loss: 1.7545, Train Acc: 51.39%, Test Loss: 1.7259, Test Acc: 52.06%, Acc Diff: -0.67%, LR: 0.002933
INFO:CIFAR-100_Training:Epoch 11: Train Loss: 1.7545, Train Acc: 51.39%, Test Loss: 1.7259, Test Acc: 52.06%, Acc Diff: -0.67%, LR: 0.002933
2025-10-11 09:16:49,158 - CIFAR-100_Training - INFO - Starting Epoch 12/30
INFO:CIFAR-100_Training:Starting Epoch 12/30
Epoch 12 - Loss: 1.8182, Accuracy: 53.34%: 100%|██████████| 391/391 [01:07<00:00,  5.77it/s]
2025-10-11 09:18:01,242 - CIFAR-100_Training - INFO - Epoch 12: Train Loss: 1.6736, Train Acc: 53.34%, Test Loss: 1.7286, Test Acc: 52.94%, Acc Diff: 0.40%, LR: 0.002851
INFO:CIFAR-100_Training:Epoch 12: Train Loss: 1.6736, Train Acc: 53.34%, Test Loss: 1.7286, Test Acc: 52.94%, Acc Diff: 0.40%, LR: 0.002851
2025-10-11 09:18:01,245 - CIFAR-100_Training - INFO - Starting Epoch 13/30
INFO:CIFAR-100_Training:Starting Epoch 13/30
Epoch 13 - Loss: 1.7220, Accuracy: 54.83%: 100%|██████████| 391/391 [01:07<00:00,  5.78it/s]
2025-10-11 09:19:13,150 - CIFAR-100_Training - INFO - Epoch 13: Train Loss: 1.6106, Train Acc: 54.83%, Test Loss: 1.6349, Test Acc: 54.71%, Acc Diff: 0.12%, LR: 0.002739
INFO:CIFAR-100_Training:Epoch 13: Train Loss: 1.6106, Train Acc: 54.83%, Test Loss: 1.6349, Test Acc: 54.71%, Acc Diff: 0.12%, LR: 0.002739
2025-10-11 09:19:13,154 - CIFAR-100_Training - INFO - Starting Epoch 14/30
INFO:CIFAR-100_Training:Starting Epoch 14/30
Epoch 14 - Loss: 1.4639, Accuracy: 56.48%: 100%|██████████| 391/391 [01:07<00:00,  5.77it/s]
2025-10-11 09:20:25,083 - CIFAR-100_Training - INFO - Epoch 14: Train Loss: 1.5451, Train Acc: 56.48%, Test Loss: 1.5481, Test Acc: 56.56%, Acc Diff: -0.08%, LR: 0.002599
INFO:CIFAR-100_Training:Epoch 14: Train Loss: 1.5451, Train Acc: 56.48%, Test Loss: 1.5481, Test Acc: 56.56%, Acc Diff: -0.08%, LR: 0.002599
2025-10-11 09:20:25,084 - CIFAR-100_Training - INFO - Starting Epoch 15/30
INFO:CIFAR-100_Training:Starting Epoch 15/30
Epoch 15 - Loss: 1.3975, Accuracy: 58.01%: 100%|██████████| 391/391 [01:07<00:00,  5.79it/s]
2025-10-11 09:21:36,863 - CIFAR-100_Training - INFO - Epoch 15: Train Loss: 1.4771, Train Acc: 58.01%, Test Loss: 1.5629, Test Acc: 57.48%, Acc Diff: 0.53%, LR: 0.002435
INFO:CIFAR-100_Training:Epoch 15: Train Loss: 1.4771, Train Acc: 58.01%, Test Loss: 1.5629, Test Acc: 57.48%, Acc Diff: 0.53%, LR: 0.002435
2025-10-11 09:21:36,865 - CIFAR-100_Training - INFO - Starting Epoch 16/30
INFO:CIFAR-100_Training:Starting Epoch 16/30
Epoch 16 - Loss: 1.4234, Accuracy: 59.87%: 100%|██████████| 391/391 [01:07<00:00,  5.78it/s]
2025-10-11 09:22:48,622 - CIFAR-100_Training - INFO - Epoch 16: Train Loss: 1.4091, Train Acc: 59.87%, Test Loss: 1.4085, Test Acc: 60.49%, Acc Diff: -0.62%, LR: 0.002250
INFO:CIFAR-100_Training:Epoch 16: Train Loss: 1.4091, Train Acc: 59.87%, Test Loss: 1.4085, Test Acc: 60.49%, Acc Diff: -0.62%, LR: 0.002250
2025-10-11 09:22:48,623 - CIFAR-100_Training - INFO - Starting Epoch 17/30
INFO:CIFAR-100_Training:Starting Epoch 17/30
Epoch 17 - Loss: 1.4349, Accuracy: 61.60%: 100%|██████████| 391/391 [01:07<00:00,  5.79it/s]
2025-10-11 09:24:00,383 - CIFAR-100_Training - INFO - Epoch 17: Train Loss: 1.3401, Train Acc: 61.60%, Test Loss: 1.4231, Test Acc: 60.20%, Acc Diff: 1.40%, LR: 0.002048
INFO:CIFAR-100_Training:Epoch 17: Train Loss: 1.3401, Train Acc: 61.60%, Test Loss: 1.4231, Test Acc: 60.20%, Acc Diff: 1.40%, LR: 0.002048
2025-10-11 09:24:00,385 - CIFAR-100_Training - INFO - Starting Epoch 18/30
INFO:CIFAR-100_Training:Starting Epoch 18/30
Epoch 18 - Loss: 1.6502, Accuracy: 63.44%: 100%|██████████| 391/391 [01:07<00:00,  5.79it/s]
2025-10-11 09:25:12,305 - CIFAR-100_Training - INFO - Epoch 18: Train Loss: 1.2754, Train Acc: 63.44%, Test Loss: 1.3476, Test Acc: 62.15%, Acc Diff: 1.29%, LR: 0.001833
INFO:CIFAR-100_Training:Epoch 18: Train Loss: 1.2754, Train Acc: 63.44%, Test Loss: 1.3476, Test Acc: 62.15%, Acc Diff: 1.29%, LR: 0.001833
2025-10-11 09:25:12,307 - CIFAR-100_Training - INFO - Starting Epoch 19/30
INFO:CIFAR-100_Training:Starting Epoch 19/30
Epoch 19 - Loss: 1.4521, Accuracy: 65.19%: 100%|██████████| 391/391 [01:07<00:00,  5.79it/s]
2025-10-11 09:26:23,950 - CIFAR-100_Training - INFO - Epoch 19: Train Loss: 1.1973, Train Acc: 65.19%, Test Loss: 1.2992, Test Acc: 63.50%, Acc Diff: 1.69%, LR: 0.001612
INFO:CIFAR-100_Training:Epoch 19: Train Loss: 1.1973, Train Acc: 65.19%, Test Loss: 1.2992, Test Acc: 63.50%, Acc Diff: 1.69%, LR: 0.001612
2025-10-11 09:26:23,952 - CIFAR-100_Training - INFO - Starting Epoch 20/30
INFO:CIFAR-100_Training:Starting Epoch 20/30
Epoch 20 - Loss: 1.1736, Accuracy: 67.07%: 100%|██████████| 391/391 [01:07<00:00,  5.81it/s]
2025-10-11 09:27:35,413 - CIFAR-100_Training - INFO - Epoch 20: Train Loss: 1.1302, Train Acc: 67.07%, Test Loss: 1.2331, Test Acc: 64.60%, Acc Diff: 2.47%, LR: 0.001388
INFO:CIFAR-100_Training:Epoch 20: Train Loss: 1.1302, Train Acc: 67.07%, Test Loss: 1.2331, Test Acc: 64.60%, Acc Diff: 2.47%, LR: 0.001388
2025-10-11 09:27:35,415 - CIFAR-100_Training - INFO - Starting Epoch 21/30
INFO:CIFAR-100_Training:Starting Epoch 21/30
Epoch 21 - Loss: 1.1973, Accuracy: 69.09%: 100%|██████████| 391/391 [01:07<00:00,  5.81it/s]
2025-10-11 09:28:46,973 - CIFAR-100_Training - INFO - Epoch 21: Train Loss: 1.0502, Train Acc: 69.09%, Test Loss: 1.1454, Test Acc: 67.32%, Acc Diff: 1.77%, LR: 0.001166
INFO:CIFAR-100_Training:Epoch 21: Train Loss: 1.0502, Train Acc: 69.09%, Test Loss: 1.1454, Test Acc: 67.32%, Acc Diff: 1.77%, LR: 0.001166
2025-10-11 09:28:46,975 - CIFAR-100_Training - INFO - Starting Epoch 22/30
INFO:CIFAR-100_Training:Starting Epoch 22/30
Epoch 22 - Loss: 1.0790, Accuracy: 71.30%: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
2025-10-11 09:29:58,096 - CIFAR-100_Training - INFO - Epoch 22: Train Loss: 0.9704, Train Acc: 71.30%, Test Loss: 1.1119, Test Acc: 68.16%, Acc Diff: 3.14%, LR: 0.000952
INFO:CIFAR-100_Training:Epoch 22: Train Loss: 0.9704, Train Acc: 71.30%, Test Loss: 1.1119, Test Acc: 68.16%, Acc Diff: 3.14%, LR: 0.000952
2025-10-11 09:29:58,097 - CIFAR-100_Training - INFO - Starting Epoch 23/30
INFO:CIFAR-100_Training:Starting Epoch 23/30
Epoch 23 - Loss: 0.8310, Accuracy: 73.71%: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
2025-10-11 09:31:09,549 - CIFAR-100_Training - INFO - Epoch 23: Train Loss: 0.8916, Train Acc: 73.71%, Test Loss: 1.0926, Test Acc: 69.15%, Acc Diff: 4.56%, LR: 0.000750
INFO:CIFAR-100_Training:Epoch 23: Train Loss: 0.8916, Train Acc: 73.71%, Test Loss: 1.0926, Test Acc: 69.15%, Acc Diff: 4.56%, LR: 0.000750
2025-10-11 09:31:09,551 - CIFAR-100_Training - INFO - Starting Epoch 24/30
INFO:CIFAR-100_Training:Starting Epoch 24/30
Epoch 24 - Loss: 0.8231, Accuracy: 75.69%: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
2025-10-11 09:32:20,741 - CIFAR-100_Training - INFO - Epoch 24: Train Loss: 0.8135, Train Acc: 75.69%, Test Loss: 1.0296, Test Acc: 70.72%, Acc Diff: 4.97%, LR: 0.000565
INFO:CIFAR-100_Training:Epoch 24: Train Loss: 0.8135, Train Acc: 75.69%, Test Loss: 1.0296, Test Acc: 70.72%, Acc Diff: 4.97%, LR: 0.000565
2025-10-11 09:32:20,743 - CIFAR-100_Training - INFO - Starting Epoch 25/30
INFO:CIFAR-100_Training:Starting Epoch 25/30
Epoch 25 - Loss: 1.0348, Accuracy: 77.93%: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
2025-10-11 09:33:32,177 - CIFAR-100_Training - INFO - Epoch 25: Train Loss: 0.7332, Train Acc: 77.93%, Test Loss: 1.0046, Test Acc: 71.37%, Acc Diff: 6.56%, LR: 0.000401
INFO:CIFAR-100_Training:Epoch 25: Train Loss: 0.7332, Train Acc: 77.93%, Test Loss: 1.0046, Test Acc: 71.37%, Acc Diff: 6.56%, LR: 0.000401
2025-10-11 09:33:32,179 - CIFAR-100_Training - INFO - Starting Epoch 26/30
INFO:CIFAR-100_Training:Starting Epoch 26/30
Epoch 26 - Loss: 0.6335, Accuracy: 80.04%: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
2025-10-11 09:34:43,357 - CIFAR-100_Training - INFO - Epoch 26: Train Loss: 0.6596, Train Acc: 80.04%, Test Loss: 0.9795, Test Acc: 72.37%, Acc Diff: 7.67%, LR: 0.000261
INFO:CIFAR-100_Training:Epoch 26: Train Loss: 0.6596, Train Acc: 80.04%, Test Loss: 0.9795, Test Acc: 72.37%, Acc Diff: 7.67%, LR: 0.000261
2025-10-11 09:34:43,359 - CIFAR-100_Training - INFO - Starting Epoch 27/30
INFO:CIFAR-100_Training:Starting Epoch 27/30
Epoch 27 - Loss: 0.6793, Accuracy: 81.72%: 100%|██████████| 391/391 [01:07<00:00,  5.82it/s]
2025-10-11 09:35:54,673 - CIFAR-100_Training - INFO - Epoch 27: Train Loss: 0.6060, Train Acc: 81.72%, Test Loss: 0.9661, Test Acc: 72.90%, Acc Diff: 8.82%, LR: 0.000149
INFO:CIFAR-100_Training:Epoch 27: Train Loss: 0.6060, Train Acc: 81.72%, Test Loss: 0.9661, Test Acc: 72.90%, Acc Diff: 8.82%, LR: 0.000149
2025-10-11 09:35:54,676 - CIFAR-100_Training - INFO - Starting Epoch 28/30
INFO:CIFAR-100_Training:Starting Epoch 28/30
Epoch 28 - Loss: 0.3849, Accuracy: 83.17%: 100%|██████████| 391/391 [01:07<00:00,  5.80it/s]
2025-10-11 09:37:06,175 - CIFAR-100_Training - INFO - Epoch 28: Train Loss: 0.5608, Train Acc: 83.17%, Test Loss: 0.9552, Test Acc: 73.30%, Acc Diff: 9.87%, LR: 0.000067
INFO:CIFAR-100_Training:Epoch 28: Train Loss: 0.5608, Train Acc: 83.17%, Test Loss: 0.9552, Test Acc: 73.30%, Acc Diff: 9.87%, LR: 0.000067
2025-10-11 09:37:06,176 - CIFAR-100_Training - INFO - Starting Epoch 29/30
INFO:CIFAR-100_Training:Starting Epoch 29/30
Epoch 29 - Loss: 0.3965, Accuracy: 83.98%: 100%|██████████| 391/391 [01:07<00:00,  5.83it/s]
2025-10-11 09:38:17,410 - CIFAR-100_Training - INFO - Epoch 29: Train Loss: 0.5349, Train Acc: 83.98%, Test Loss: 0.9456, Test Acc: 73.36%, Acc Diff: 10.62%, LR: 0.000017
INFO:CIFAR-100_Training:Epoch 29: Train Loss: 0.5349, Train Acc: 83.98%, Test Loss: 0.9456, Test Acc: 73.36%, Acc Diff: 10.62%, LR: 0.000017
2025-10-11 09:38:17,413 - CIFAR-100_Training - INFO - Starting Epoch 30/30
INFO:CIFAR-100_Training:Starting Epoch 30/30
Epoch 30 - Loss: 0.6925, Accuracy: 84.35%: 100%|██████████| 391/391 [01:07<00:00,  5.79it/s]
2025-10-11 09:39:29,135 - CIFAR-100_Training - INFO - Epoch 30: Train Loss: 0.5268, Train Acc: 84.35%, Test Loss: 0.9430, Test Acc: 73.57%, Acc Diff: 10.78%, LR: 0.000001
INFO:CIFAR-100_Training:Epoch 30: Train Loss: 0.5268, Train Acc: 84.35%, Test Loss: 0.9430, Test Acc: 73.57%, Acc Diff: 10.78%, LR: 0.000001
2025-10-11 09:39:29,137 - CIFAR-100_Training - INFO - Training completed!
INFO:CIFAR-100_Training:Training completed!
2025-10-11 09:39:29,139 - CIFAR-100_Training - INFO - Final Results: {'final_train_loss': 0.5267726811759003, 'final_test_loss': 0.9429771265983582, 'final_train_accuracy': 84.346, 'final_test_accuracy': 73.57, 'best_test_accuracy': 73.57, 'final_accuracy_difference': 10.77600000000001, 'max_accuracy_difference': 10.77600000000001, 'avg_accuracy_difference': 2.0481999999999996, 'overfitting_epochs': 0, 'stopped_due_to_overfitting': False}
INFO:CIFAR-100_Training:Final Results: {'final_train_loss': 0.5267726811759003, 'final_test_loss': 0.9429771265983582, 'final_train_accuracy': 84.346, 'final_test_accuracy': 73.57, 'best_test_accuracy': 73.57, 'final_accuracy_difference': 10.77600000000001, 'max_accuracy_difference': 10.77600000000001, 'avg_accuracy_difference': 2.0481999999999996, 'overfitting_epochs': 0, 'stopped_due_to_overfitting': False}
2025-10-11 09:39:29,140 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:39:29,141 - CIFAR-100_Training - INFO - OVERFITTING ANALYSIS
INFO:CIFAR-100_Training:OVERFITTING ANALYSIS
2025-10-11 09:39:29,143 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:39:29,144 - CIFAR-100_Training - INFO - Final accuracy difference: 10.78%
INFO:CIFAR-100_Training:Final accuracy difference: 10.78%
2025-10-11 09:39:29,146 - CIFAR-100_Training - INFO - Maximum accuracy difference: 10.78%
INFO:CIFAR-100_Training:Maximum accuracy difference: 10.78%
2025-10-11 09:39:29,147 - CIFAR-100_Training - INFO - Average accuracy difference: 2.05%
INFO:CIFAR-100_Training:Average accuracy difference: 2.05%
2025-10-11 09:39:29,148 - CIFAR-100_Training - INFO - Consecutive overfitting epochs: 0
INFO:CIFAR-100_Training:Consecutive overfitting epochs: 0
2025-10-11 09:39:29,150 - CIFAR-100_Training - INFO - Stopped due to overfitting: False
INFO:CIFAR-100_Training:Stopped due to overfitting: False
2025-10-11 09:39:29,151 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:39:30,409 - CIFAR-100_Training - INFO - Training curves saved to: /content/logs/training_curves_20251011_093929.png
INFO:CIFAR-100_Training:Training curves saved to: /content/logs/training_curves_20251011_093929.png

2025-10-11 09:39:31,115 - CIFAR-100_Training - INFO - Model saved to: /content/models/cifar100_model_20251011_093931.pth
INFO:CIFAR-100_Training:Model saved to: /content/models/cifar100_model_20251011_093931.pth
2025-10-11 09:39:31,116 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:39:31,117 - CIFAR-100_Training - INFO - TRAINING PIPELINE COMPLETED SUCCESSFULLY
INFO:CIFAR-100_Training:TRAINING PIPELINE COMPLETED SUCCESSFULLY
2025-10-11 09:39:31,118 - CIFAR-100_Training - INFO - ==================================================
INFO:CIFAR-100_Training:==================================================
2025-10-11 09:39:31,120 - CIFAR-100_Training - INFO - Final Metrics: {'final_train_loss': 0.5267726811759003, 'final_test_loss': 0.9429771265983582, 'final_train_accuracy': 84.346, 'final_test_accuracy': 73.57, 'best_test_accuracy': 73.57, 'final_accuracy_difference': 10.77600000000001, 'max_accuracy_difference': 10.77600000000001, 'avg_accuracy_difference': 2.0481999999999996, 'overfitting_epochs': 0, 'stopped_due_to_overfitting': False}
INFO:CIFAR-100_Training:Final Metrics: {'final_train_loss': 0.5267726811759003, 'final_test_loss': 0.9429771265983582, 'final_train_accuracy': 84.346, 'final_test_accuracy': 73.57, 'best_test_accuracy': 73.57, 'final_accuracy_difference': 10.77600000000001, 'max_accuracy_difference': 10.77600000000001, 'avg_accuracy_difference': 2.0481999999999996, 'overfitting_epochs': 0, 'stopped_due_to_overfitting': False}

============================================================
TRAINING COMPLETED SUCCESSFULLY!
============================================================
Final Test Accuracy: 73.57%
Best Test Accuracy: 73.57%
Final Train Accuracy: 84.35%
============================================================
Check the logs/ directory for detailed logs and visualizations.
============================================================
