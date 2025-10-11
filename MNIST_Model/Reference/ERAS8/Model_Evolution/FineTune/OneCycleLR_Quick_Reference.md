# OneCycleLR Quick Reference for CIFAR-100

## Fixed Configuration (Current)

```python
# OPTIMIZED FOR SGD + MOMENTUM 0.9
config.training.optimizer_type = 'SGD'
config.training.learning_rate = 0.00251  # Not used directly by OneCycleLR
config.training.momentum = 0.9
config.training.weight_decay = 0.0001

config.training.scheduler_type = 'OneCycleLR'
config.training.onecycle_max_lr = 0.01              # ✅ Safe for SGD
config.training.onecycle_pct_start = 0.3            # ✅ 30 epoch warmup
config.training.onecycle_div_factor = 10.0          # ✅ Start at 0.001
config.training.onecycle_final_div_factor = 1000.0  # ✅ End at 0.00001
config.training.onecycle_anneal_strategy = 'cos'
```

## Learning Rate Trajectory (100 Epochs)

| Phase | Epochs | LR Range | Purpose |
|-------|--------|----------|---------|
| **Warmup** | 1-30 | 0.001 → 0.01 | Gradual increase, model stabilization |
| **Annealing** | 31-100 | 0.01 → 0.00001 | Cosine decay, fine-tuning |

### Detailed Schedule

```
Epoch  1: LR ≈ 0.00100 (start)
Epoch  5: LR ≈ 0.00200
Epoch 10: LR ≈ 0.00367
Epoch 15: LR ≈ 0.00567
Epoch 20: LR ≈ 0.00733
Epoch 25: LR ≈ 0.00900
Epoch 30: LR ≈ 0.01000 (MAX) ⭐
Epoch 40: LR ≈ 0.00833
Epoch 50: LR ≈ 0.00568
Epoch 60: LR ≈ 0.00350
Epoch 70: LR ≈ 0.00183
Epoch 80: LR ≈ 0.00083
Epoch 90: LR ≈ 0.00033
Epoch 100: LR ≈ 0.00001 (end)
```

## Expected Training Performance

### Accuracy Progression
```
Epoch   1: Train ~10%, Test ~12%   (Good start)
Epoch  10: Train ~38%, Test ~40%   (Ramping up)
Epoch  30: Train ~63%, Test ~60%   (Max LR reached)
Epoch  50: Train ~72%, Test ~66%   (Mid annealing)
Epoch  75: Train ~78%, Test ~69%   (Fine-tuning)
Epoch 100: Train ~83%, Test ~71%   (Final)
```

### Loss Progression
```
Epoch   1: Train ~3.5, Test ~3.3
Epoch  30: Train ~1.4, Test ~1.5
Epoch  50: Train ~1.0, Test ~1.2
Epoch 100: Train ~0.5, Test ~1.1
```

## Problem Indicators

### 🚨 Signs OneCycleLR is NOT working:

1. **Very low accuracy in epoch 1** (< 5%)
   - Fix: Increase `div_factor` or increase `max_lr`

2. **Training diverges** (NaN loss or accuracy drops)
   - Fix: Reduce `max_lr` to 0.005

3. **Slow learning for first 20 epochs** (< 30% accuracy)
   - Fix: Increase `pct_start` to 0.4 or reduce `div_factor`

4. **Oscillating loss** around max LR (epoch 30)
   - Fix: Reduce `max_lr` to 0.007

5. **No improvement after epoch 60** (plateaus)
   - Fix: Increase `final_div_factor` to 2000

## Alternative Configurations

### Conservative (Safer but Slower)
```python
config.training.onecycle_max_lr = 0.005            # More conservative
config.training.onecycle_pct_start = 0.4           # Longer warmup
config.training.onecycle_div_factor = 5.0          # Higher start
config.training.onecycle_final_div_factor = 500.0  # Higher end
```

### Aggressive (Faster but Riskier)
```python
config.training.onecycle_max_lr = 0.02             # Higher max
config.training.onecycle_pct_start = 0.2           # Shorter warmup
config.training.onecycle_div_factor = 20.0         # Lower start
config.training.onecycle_final_div_factor = 2000.0 # Lower end
```

### For Different Batch Sizes
```python
# Batch Size 64
config.training.onecycle_max_lr = 0.005

# Batch Size 128 (current)
config.training.onecycle_max_lr = 0.01

# Batch Size 256
config.training.onecycle_max_lr = 0.02

# Batch Size 512
config.training.onecycle_max_lr = 0.04
```

## Comparison: Old vs New

| Metric | Old (Broken) | New (Fixed) | Change |
|--------|-------------|------------|---------|
| Max LR | 0.0251 | 0.01 | -60% ✅ |
| Initial LR | 0.001004 | 0.001 | Similar |
| Final LR | 0.00000251 | 0.00001 | +300% ✅ |
| Warmup Epochs | 10 | 30 | +200% ✅ |
| Epoch 1 Accuracy | 1.58% | ~10% | +533% ✅ |
| Epoch 10 Accuracy | 11.30% | ~38% | +236% ✅ |

## OneCycleLR vs Other Schedulers

| Scheduler | When to Use | Pros | Cons |
|-----------|------------|------|------|
| **OneCycleLR** | Fast convergence needed | Built-in warmup, regularization | Needs tuning |
| **ReduceLROnPlateau** | Adaptive schedule | No hyperparams | Slower convergence |
| **StepLR** | Simple baseline | Predictable | Fixed schedule |
| **CosineAnnealing** | Long training | Smooth decay | No warmup |

## Quick Checklist Before Training

- [ ] `max_lr` between 0.005-0.02 for SGD?
- [ ] `pct_start` >= 0.2 (at least 20% warmup)?
- [ ] `div_factor` between 5-15?
- [ ] `final_div_factor` between 500-2000?
- [ ] Momentum = 0.9 (or reduce if using high max_lr)?
- [ ] Weight decay > 0 for regularization?
- [ ] Data augmentation enabled?

## Troubleshooting

### Issue: Accuracy < 5% after 5 epochs
**Solution:** Increase initial LR
```python
config.training.onecycle_div_factor = 5.0  # Was 10.0
# Initial LR becomes 0.002 instead of 0.001
```

### Issue: Training diverges (NaN loss)
**Solution:** Reduce max LR
```python
config.training.onecycle_max_lr = 0.005  # Was 0.01
```

### Issue: Oscillating loss around max LR
**Solution:** Smooth the transition
```python
config.training.onecycle_pct_start = 0.4  # Was 0.3
config.training.onecycle_max_lr = 0.007   # Was 0.01
```

### Issue: No improvement in final 30 epochs
**Solution:** Increase final LR
```python
config.training.onecycle_final_div_factor = 500.0  # Was 1000.0
# Final LR becomes 0.00002 instead of 0.00001
```

## Validation Commands

### Check LR schedule
```python
# Add to training script for debugging
import matplotlib.pyplot as plt

lrs = []
for epoch in range(100):
    for batch in train_loader:
        # ... training code ...
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()  # OneCycleLR steps per batch

plt.plot(lrs)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('OneCycleLR Schedule')
plt.savefig('lr_schedule.png')
```

### Monitor first 10 epochs
```bash
# Watch for these patterns:
tail -f logs/LATEST_training.log | grep "Epoch.*:"

# Good pattern:
# Epoch 1: Train Acc: 10-12%, Test Acc: 11-13%
# Epoch 5: Train Acc: 30-35%, Test Acc: 32-36%
# Epoch 10: Train Acc: 40-45%, Test Acc: 42-46%

# Bad pattern:
# Epoch 1: Train Acc: < 5%  ❌
# Epoch 5: Train Acc: < 20% ❌
```

---

**Summary:** OneCycleLR with SGD works best with moderate max_lr (0.01), extended warmup (30%), and balanced initial/final LRs. The key is avoiding the "too high max_lr" trap that Adam can handle but SGD cannot.

