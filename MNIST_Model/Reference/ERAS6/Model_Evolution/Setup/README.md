# MNIST Training Experiments

## Experiment Summary Table

| Experiment | Date | Architecture | Parameters | Train Acc | Test Acc | Gap | Epochs | LR | Batch Size | Status |
|------------|------|--------------|------------|-----------|----------|-----|--------|----|-----------|---------| 
| Setup-1 | 2025-09-24 | Rough CNN Setup | 6,379,786 | 99.94% | 99.52% | 0.42% | 15 | 0.1 | 64 | ✅ Completed |
| Setup-2 | 2025-09-25 | Cake Architecture | 388,582 | 99.81% | 99.33% | 0.48% | 20 | 0.01 | 64 | ✅ Completed |

---

## Experiment Details

### Setup-2: Cake Architecture

#### Target
| Objective | Status |
|-----------|--------|
| Set up model skeleton with widely used cake architecture structure | ✅ |
| Reduce parameters with no major loss in accuracy | ✅ |
| Achieve stable training  | ✅ |
| Monitor for overfitting vs underfitting | ✅ |
| Reduce parameters to ~8K target | ❌ Still 388K (need further reduction) |

#### Model Architecture
| Layer | Type | Input | Output | Kernel | Padding | Parameters |
|-------|------|-------|--------|--------|---------|------------|
| ConvBlock1 | Conv2d + ReLU | 1 | 16 | 3x3 | 1 | 144 |
| ConvBlock2 | Conv2d + ReLU | 16 | 16 | 3x3 | 1 | 2,304 |
| ConvBlock3 | Conv2d + ReLU | 16 | 16 | 3x3 | 1 | 2,304 |
| Pool1 | MaxPool2d | - | - | 2x2 | 0 | 0 |
| Dropout1 | Dropout | - | - | - | - | 0 |
| ConvBlock4 | Conv2d + ReLU | 16 | 32 | 3x3 | 1 | 4,608 |
| ConvBlock5 | Conv2d + ReLU | 32 | 32 | 3x3 | 1 | 9,216 |
| Pool2 | MaxPool2d | - | - | 2x2 | 0 | 0 |
| ConvBlock6 | Conv2d + ReLU | 32 | 64 | 3x3 | 1 | 18,432 |
| ConvBlock7 | Conv2d + ReLU | 64 | 64 | 3x3 | 1 | 36,864 |
| FC0 | Linear | 3136 | 100 | - | - | 313,700 |
| FC1 | Linear | 100 | 10 | - | - | 1,010 |
| **Total** | | | | | | **388,582** |

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Learning Rate | 0.01 |
| Batch Size | 64 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight Decay | 0.0 |
| Scheduler | StepLR |
| Scheduler Step Size | 10 |
| Scheduler Gamma | 0.1 |
| Device | CUDA 12.6 |

#### Results
| Metric | Value |
|--------|-------|
| Total Parameters | 388,582 |
| Best Training Accuracy | 99.83% |
| Best Test Accuracy | 99.41% |
| Final Training Accuracy | 99.81% |
| Final Test Accuracy | 99.33% |
| Training Time | ~7 minutes |
| Overfitting Gap | 0.48% |

#### Key Observations
| Aspect | Observation | Analysis |
|--------|-------------|----------|
| **Architecture Design** | Cake architecture with nn.Sequential blocks | ✅ Well-structured, modular design |
| **Parameter Reduction** | 94% reduction from 6.38M to 388K parameters | ✅ Significant improvement |
| **Training Stability** | Fixed NaN loss issue with proper LR and BN | ✅ Stable training achieved |
| **Overfitting** | Slight overfitting (0.48% gap) | ✅ Good generalization |
| **Test Accuracy Stability** | Stable but slight dip at end (99.41% → 99.33%) | ⚠️ Minor performance degradation |
| **Convergence** | Train accuracy plateaued after epoch 10 | ⚠️ Suggests need for capacity improvement |
| **FC Layer Dominance** | FC0 layer consumes 80.7% of parameters (313,700/388,582) | ⚠️ Major inefficiency identified |
| **Parameter Target** | Still 48x larger than 8K target | ❌ Need major architecture reduction |

#### Critical Issues Identified
| Issue | Impact | Priority | Solution Needed |
|-------|--------|----------|-----------------|
| **FC Parameter Dominance** | 80.7% of parameters in FC0 layer alone | 🔴 Critical | Replace with GAP + conv-to-class |
| **Parameter Count** | 388K vs 8K target (48x too large) | 🔴 Critical | Major architecture reduction needed |
| **Capacity Limitation** | Post-epoch 10 plateau | �� Medium | Enhance final output blocks |
| **Test Accuracy Dip** | 0.08% performance degradation | �� Low | Minor concern |

#### Parameter Distribution Analysis
| Layer Type | Parameters | Percentage | Efficiency |
|------------|------------|------------|------------|
| **Convolutional** | 74,872 | 19.3% | ✅ Efficient |
| **FC0 (3136→100)** | 313,700 | 80.7% | ❌ Very inefficient |
| **FC1 (100→10)** | 1,010 | 0.3% | ✅ Efficient |
| **Total** | 388,582 | 100% | ❌ FC-dominated |

#### Improvements Needed
| Improvement | Current Issue | Target Solution |
|-------------|---------------|-----------------|
| **Replace FC0 Layer** | 3136→100 (313,700 params) | Use Global Average Pooling |
| **Eliminate FC1 Layer** | 100→10 (1,010 params) | Direct conv-to-class mapping |
| **Reduce Channel Dimensions** | 64 channels in final conv | Reduce to 10-16 channels |
| **Architecture Efficiency** | Total: 388K parameters | Target: <8K total parameters |

#### Next Steps
| Action | Priority | Reason |
|--------|----------|--------|
| Replace FC layers with GAP | 🔴 Critical | FC0 alone consumes 80.7% of parameters |
| Implement conv-to-class mapping | 🔴 Critical | Eliminate FC layer inefficiency |
| Reduce final conv channels | 🟡 High | Further parameter reduction needed |
| Test 8K parameter target | 🟡 High | Verify if target is achievable |
| Analyze parameter distribution | ✅ Completed | FC dominance clearly identified |

---

### Setup-1: Rough CNN Setup

#### Target
| Objective | Status |
|-----------|--------|
| Get UV package management and CUDA 12.6 setup | ✅ |
| Set up proper data transforms with augmentation | ✅ |
| Set up flexible data loaders with dynamic input size | ✅ |
| Set up basic working code with OOP design patterns | ✅ |
| Set up comprehensive training & test loop with logging | ✅ |
| Implement script-relative logging and model saving | ✅ |
| Add detailed model summary logging | ✅ |
| Test Rough CNN Setup  | ✅ |

#### Model Architecture
| Layer | Type | Input | Output | Kernel | Padding | Parameters |
|-------|------|-------|--------|--------|---------|------------|
| Conv1 | Conv2d | 1 | 32 | 3x3 | 1 | 320 |
| Conv2 | Conv2d | 32 | 64 | 3x3 | 1 | 18,496 |
| Pool1 | MaxPool2d | - | - | 2x2 | 0 | 0 |
| Conv3 | Conv2d | 64 | 128 | 3x3 | 1 | 73,856 |
| Conv4 | Conv2d | 128 | 256 | 3x3 | 1 | 295,168 |
| Pool2 | MaxPool2d | - | - | 2x2 | 0 | 0 |
| Conv5 | Conv2d | 256 | 512 | 3x3 | 0 | 1,180,160 |
| Conv6 | Conv2d | 512 | 1024 | 3x3 | 0 | 4,719,616 |
| Conv7 | Conv2d | 1024 | 10 | 3x3 | 0 | 92,170 |

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Learning Rate | 0.1 |
| Batch Size | 64 |
| Optimizer | SGD |
| Momentum | 0.8 |
| Weight Decay | 0.0 |
| Scheduler | StepLR |
| Scheduler Step Size | 10 |
| Scheduler Gamma | 0.1 |
| Device | CUDA 12.6 |

#### Results
| Metric | Value |
|--------|-------|
| Total Parameters | 6,379,786 |
| Best Training Accuracy | 99.94% |
| Best Test Accuracy | 99.52% |
| Final Training Accuracy | 99.94% |
| Final Test Accuracy | 99.47% |
| Training Time | ~9 minutes |
| Overfitting Gap | 0.47% |

#### Analysis
| Aspect | Observation |
|--------|-------------|
| Architecture Changes | Removed batch norm, dropout, and FC layers |
| Channel Progression | 1→32→64→128→256→512→1024→10 |
| Regularization | None (no dropout or batch norm) |
| Output Method | Direct conv-to-class mapping |
| Expected Overfitting | High (no regularization) |
| Training Speed | Fast (simplified architecture) |
| Parameter Efficiency | 6.38M parameters for 99.52% accuracy |

#### Performance Analysis
| Criteria | Status | Notes |
|----------|--------|-------|
| Overfitting | ✅ Excellent | Gap = 0.47% (<0.5% = Good generalization) |
| Generalization | ✅ Excellent | Very small train-test gap |
| Training Stability | ✅ Excellent | Smooth convergence, no oscillations |
| Convergence Speed | ✅ Fast | Best accuracy reached by epoch 12 |

#### Next Steps
| Action | Priority | Reason |
|--------|----------|--------|
| ~~Monitor overfitting~~ | ✅ Completed | No overfitting detected (0.47% gap) |
| Compare with baseline | Medium | Need reference for improvement |
| ~~Add regularization~~ | ❌ Not needed | No overfitting, model generalizes well |
| Analyze parameter efficiency | Medium | 6.38M params for 99.52% accuracy |
| Document lessons learned | ✅ Completed | Excellent results with simplified architecture. But model size is too high. Will try to reduce model size |

---

## Experiment Comparison Table

| Metric | Setup-1 | [Future Exp] | [Future Exp] | Notes |
|--------|---------|--------------|--------------|-------|
| Parameters | 6,379,786 | - | - | Very high parameter count |
| Train Acc | 99.94% | - | - | Excellent training performance |
| Test Acc | 99.52% | - | - | Excellent generalization |
| Gap | 0.47% | - | - | Minimal overfitting |
| Training Time | ~9 min | - | - | Fast training on CUDA |
| Architecture | Simplified CNN | - | - | No regularization needed |

---

## Technical Implementation

### Key Features
| Feature | Implementation | Status |
|---------|----------------|--------|
| OOP Design | Design patterns (Singleton, Factory, Strategy, etc.) | ✅ |
| Logging | Script-relative file paths with timestamps | ✅ |
| Dynamic Input Size | Auto-detection from data loader | ✅ |
| Configuration | Dataclasses with flexible settings | ✅ |
| CUDA Support | Automatic device detection | ✅ |
| Model Persistence | Auto-save with timestamps | ✅ |

### Usage Commands
| Command | Description |
|---------|-------------|
| `uv run python mnist_training.py` | Run training from script directory |
| `uv run python ERAS6/Model_Evolution/Setup/mnist_training.py` | Run from project root |

---

## Experiment Log

| Date | Experiment | Action | Notes |
|------|------------|--------|-------|
| 2025-09-24 | Setup-1 | Completed | Excellent results: 99.52% test accuracy, minimal overfitting |

---

## Key Insights

### What Worked Well
- **Fast Convergence**: Model reached 99%+ accuracy by epoch 2
- **Excellent Generalization**: Only 0.47% gap between train and test accuracy

### Surprising Results
- **No Overfitting**: Despite 6.38M parameters and no regularization, model generalizes well
- **Fast Training**: 9 minutes for 15 epochs on CUDA
- **High Accuracy**: 99.52% test accuracy with simplified architecture

### Lessons Learned
- MNIST is simple enough that complex regularization may not be necessary
- Direct conv-to-class mapping can be very effective
- Large parameter count doesn't always lead to overfitting on simple datasets

---

*This README documents the successful completion of Setup-1 with excellent results. The simplified CNN architecture achieved 99.52% test accuracy with minimal overfitting.*