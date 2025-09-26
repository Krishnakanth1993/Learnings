# MNIST Training Experiments - Optimization Phase

## Project Summary

### Target
- ✅ Optimize model architecture for better performance
- ✅ Experiment with different regularization techniques
- ✅ Compare models with and without dropout
- ✅ Achieve consistent high accuracy with minimal parameters
- ✅ Maintain model stability and prevent overfitting

### Result
- **Best Parameters**: 6,132
- **Best Test Accuracy**: 99.34%
- **Best Train Accuracy**: 99.47%
- **Model Stability**: Excellent (minimal gap between train/test)

### Analysis
- **Excellent optimization!** Both experiments achieved >99% accuracy
- **Dropout impact**: Slight improvement in generalization (99.27% → 99.34%)
- **Consistent performance**: Both models show stable training curves
- **Ultra-lightweight**: Maintained 6K parameters while achieving high accuracy
- **Production ready**: Robust models suitable for deployment

### Resources
- **Experiment 1 Log**: [View Training Log (No Dropout)](https://github.com/Krishnakanth1993/Learnings/blob/main/MNIST_Model/Reference/ERAS6/Model_Evolution/Optimize/logs/20250926_232239_mnist_training.log)
- **Experiment 1 Curves**: [View Training Curves (No Dropout)](https://github.com/Krishnakanth1993/Learnings/blob/main/MNIST_Model/Reference/ERAS6/Model_Evolution/Optimize/logs/training_curves_20250926_233350.png)
- **Experiment 2 Log**: [View Training Log (With Dropout)](https://github.com/Krishnakanth1993/Learnings/blob/main/MNIST_Model/Reference/ERAS6/Model_Evolution/Optimize/logs/20250926_233654_mnist_training.log)
- **Experiment 2 Curves**: [View Training Curves (With Dropout)](https://github.com/Krishnakanth1993/Learnings/blob/main/MNIST_Model/Reference/ERAS6/Model_Evolution/Optimize/logs/training_curves_20250926_234555.png)

---

## Experiment Summary Table

| Experiment | Date | Architecture | Parameters | Train Acc | Test Acc | Gap | Epochs | LR | Batch Size | Dropout | Status |
|------------|------|--------------|------------|-----------|----------|-----|--------|----|-----------|---------|---------| 
| Opt-1 | 2025-09-26 | Optimized CNN (No Dropout) | 6,132 | 99.47% | 99.27% | 0.20% | 20 | 0.01 | 64 | No | ✅ Completed |
| Opt-2 | 2025-09-26 | Optimized CNN (With Dropout) | 6,132 | 98.94% | 99.33% | -0.39% | 20 | 0.01 | 64 | Yes | ✅ Completed |

---

## Experiment Details

### Opt-1: Optimized CNN (No Dropout)

#### Target
| Objective | Status |
|-----------|--------|
| Optimize model architecture | ✅ |
| Remove unnecessary complexity | ✅ |
| Achieve high accuracy with minimal parameters | ✅ |
| Test model without dropout | ✅ |

#### Model Architecture
- **Total Parameters**: 6,132
- **Architecture**: Optimized CNN with BatchNorm, no Dropout
- **Key Features**:
  - Input Block: Conv2d(1→10) + BatchNorm + ReLU
  - Conv Block 1: Conv2d(10→10) + BatchNorm + ReLU
  - Conv Block 2: Conv2d(10→10) + BatchNorm + ReLU
  - Transition 1: MaxPool2d(2,2)
  - Conv Block 3: Conv2d(10→8) + BatchNorm + ReLU
  - Conv Block 4: Conv2d(8→8) + BatchNorm + ReLU
  - Transition 2: MaxPool2d(2,2)
  - Conv Block 5: Conv2d(8→8) + BatchNorm + ReLU
  - Conv Block 6: Conv2d(8→8) + BatchNorm + ReLU
  - Output: Linear(72→20) + Linear(20→10)

#### Training Configuration
- **Epochs**: 20
- **Learning Rate**: 0.01
- **Batch Size**: 64
- **Optimizer**: SGD with momentum=0.9
- **Scheduler**: StepLR (step_size=10, gamma=0.1)
- **Weight Decay**: 0.0

#### Results
- **Final Train Accuracy**: 99.47%
- **Final Test Accuracy**: 99.27%
- **Best Test Accuracy**: 99.31%
- **Final Train Loss**: 0.0163
- **Final Test Loss**: 0.0215
- **Gap**: 0.20% (minimal overfitting)

#### Key Observations
- **Excellent performance**: Achieved >99% accuracy consistently
- **Stable training**: Smooth convergence without oscillations
- **Minimal overfitting**: Very small gap between train and test accuracy
- **Fast convergence**: Reached 99%+ accuracy by epoch 10
- **Consistent results**: Stable performance across all epochs

---

### Opt-2: Optimized CNN (With Dropout)

#### Target
| Objective | Status |
|-----------|--------|
| Test dropout impact on generalization | ✅ |
| Compare with no-dropout model | ✅ |
| Achieve better test performance | ✅ |
| Maintain model stability | ✅ |

#### Model Architecture
- **Total Parameters**: 6,132 (same as Opt-1)
- **Architecture**: Optimized CNN with BatchNorm + Dropout
- **Key Features**:
  - Same architecture as Opt-1
  - **Added Dropout layers** after each conv block
  - Dropout rate: 0.05 (5%)
  - All other components identical to Opt-1

#### Training Configuration
- **Epochs**: 20
- **Learning Rate**: 0.01
- **Batch Size**: 64
- **Optimizer**: SGD with momentum=0.9
- **Scheduler**: StepLR (step_size=10, gamma=0.1)
- **Weight Decay**: 0.0
- **Dropout Rate**: 0.05

#### Results
- **Final Train Accuracy**: 98.94%
- **Final Test Accuracy**: 99.33%
- **Best Test Accuracy**: 99.34%
- **Final Train Loss**: 0.0330
- **Final Test Loss**: 0.0233
- **Gap**: -0.39% (test > train - excellent generalization!)

#### Key Observations
- **Better generalization**: Test accuracy > Train accuracy
- **Improved test performance**: 99.27% → 99.34% (+0.07%)
- **Reduced overfitting**: Negative gap indicates excellent generalization
- **Slightly slower convergence**: Dropout adds regularization effect. Need to work on LR at later stage to achieve > 99.4% within 15 epochs.
- **More stable**: Better generalization trade-off
- **Need to add more channels in initial layers yet maintain parameters less than 8K** : Planning to use GAP in next experiment

---

## Comparative Analysis

### Performance Comparison
| Metric | Opt-1 (No Dropout) | Opt-2 (With Dropout) | Improvement |
|--------|-------------------|---------------------|-------------|
| **Test Accuracy** | 99.27% | 99.34% | +0.07% |
| **Train Accuracy** | 99.47% | 98.94% | -0.53% |
| **Gap** | 0.20% | -0.39% | Better generalization |
| **Best Test** | 99.31% | 99.34% | +0.03% |
| **Parameters** | 6,132 | 6,132 | Same |

### Key Insights

#### Dropout Impact
- **Positive**: Improved test accuracy and generalization
- **Trade-off**: Slight reduction in train accuracy for better test performance
- **Stability**: Better generalization (test > train accuracy)

#### Model Efficiency
- **Parameter Count**: Maintained at 6,132 parameters
- **Architecture**: Same base structure with added regularization
- **Performance**: Both models exceed 99% accuracy

#### Training Characteristics
- **Convergence**: Both models converge smoothly
- **Stability**: No overfitting issues in either model
- **Consistency**: Reliable performance across epochs

### Recommendations

#### For Production
- **Use Opt-2 (With Dropout)**: Better generalization and test performance
- **Model Size**: 6,132 parameters (0.02 MB) - extremely lightweight
- **Accuracy**: 99.34% test accuracy - production ready

#### For Further Optimization
- **Data Augmentation**: Could potentially improve further
- **Learning Rate Scheduling**: Fine-tune for better convergence
- **Architecture**: Current design is well-optimized for MNIST

---

## Technical Specifications

### Model Architecture Details
```
Input: (1, 28, 28)
├── Conv2d(1→10, 3x3) + BatchNorm + ReLU
├── Conv2d(10→10, 3x3) + BatchNorm + ReLU [+ Dropout]
├── Conv2d(10→10, 3x3) + BatchNorm + ReLU [+ Dropout]
├── MaxPool2d(2,2)
├── Conv2d(10→8, 3x3) + BatchNorm + ReLU [+ Dropout]
├── Conv2d(8→8, 3x3) + BatchNorm + ReLU [+ Dropout]
├── MaxPool2d(2,2)
├── Conv2d(8→8, 3x3) + BatchNorm + ReLU [+ Dropout]
├── Conv2d(8→8, 3x3) + BatchNorm + ReLU [+ Dropout]
├── Flatten
├── Linear(72→20)
└── Linear(20→10)
```

### Training Environment
- **Device**: CUDA (GPU acceleration)
- **Framework**: PyTorch
- **Data**: MNIST (60K train, 10K test)
- **Transforms**: RandomRotation(-7°, 7°), ToTensor, Normalize
- **Memory**: ~0.8 MB total model size

---

## Conclusion

Both optimization experiments successfully achieved the target objectives:

1. **Maintained ultra-lightweight design** (6,132 parameters)
2. **Achieved excellent accuracy** (>99% on both models)
3. **Demonstrated model stability** (minimal overfitting)
4. **Validated regularization impact** (dropout improves generalization)

The **Opt-2 model with dropout** is recommended for production use due to its superior generalization performance and test accuracy of 99.34%.
