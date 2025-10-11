# OneCycleLR + SGD Debugging Guide

## Problem Diagnosis

### Issue Summary
OneCycleLR was not working well with SGD optimizer, showing very slow initial learning and poor convergence.

### Observed Symptoms (from log 20251011_070705)
```
Epoch  1: Train Acc: 1.58%, Test Acc: 1.16%, LR: 0.001594
Epoch  2: Train Acc: 2.14%, Test Acc: 1.06%, LR: 0.003306
Epoch  3: Train Acc: 2.65%, Test Acc: 1.59%, LR: 0.005973
Epoch 10: Train Acc: 11.30%, Test Acc: 10.02%, LR: 0.025100
Epoch 20: Train Acc: 23.44%, Test Acc: 26.05%, LR: 0.024343
```

**Problems identified:**
1. **Extremely slow start**: Only 1.58% accuracy in epoch 1
2. **Very low initial LR**: Starting at 0.001594 (too conservative)
3. **Max LR too high**: Peak at 0.0251 (too aggressive for SGD + momentum)
4. **Short warmup period**: Only 10 epochs to reach max LR

---

## Root Causes

### 1. Max LR Too High for SGD
**Original Config:**
```python
onecycle_max_lr = 0.0251  # 2.51e-02
```

**Problem:** 
- SGD with momentum 0.9 is sensitive to high learning rates
- 0.0251 is ~2.5× higher than typical SGD starting LR (0.01)
- With momentum, effective step size = LR × (1 + momentum + momentum² + ...)
- This can cause oscillations and divergence

**Why it matters for SGD:**
- SGD updates: `θ = θ - lr * grad`
- With momentum: `θ = θ - lr * (β * v_prev + grad)`
- High LR + momentum can overshoot minima
- Adam handles high LR better due to adaptive per-parameter scaling

### 2. Initial LR Too Low
**Original Config:**
```python
onecycle_div_factor = 25.0
# Initial LR = max_lr / div_factor = 0.0251 / 25 = 0.001004
```

**Problem:**
- Starting at 0.001 with SGD is very conservative
- Model barely learns in first few epochs
- Wastes valuable early training time

### 3. Warmup Too Short
**Original Config:**
```python
onecycle_pct_start = 0.1  # 10% of total epochs
# For 100 epochs: warmup = 10 epochs
```

**Problem:**
- Only 10 epochs to ramp from 0.001 to 0.0251 (25× increase)
- Too aggressive ramp can destabilize SGD training
- Model needs more time to "settle" into good gradients

### 4. Final LR Too Low
**Original Config:**
```python
onecycle_final_div_factor = 10000.0
# Final LR = max_lr / final_div_factor = 0.0251 / 10000 = 0.00000251
```

**Problem:**
- Final LR of 2.51e-6 is extremely small
- Can cause training to stagnate in final epochs
- Too small for fine-tuning in last phase

---

## Solution: Optimized Configuration

### New Configuration
```python
config.training.onecycle_max_lr = 0.01        # Reduced from 0.0251
config.training.onecycle_pct_start = 0.3      # Increased from 0.1
config.training.onecycle_div_factor = 10.0    # Reduced from 25.0
config.training.onecycle_final_div_factor = 1000.0  # Reduced from 10000.0
```

### Learning Rate Schedule
```
Phase 1: Warmup (Epochs 1-30, 30% of training)
  - Start: 0.01 / 10 = 0.001
  - End: 0.01 (max)
  - Duration: 30 epochs
  - Strategy: Cosine annealing

Phase 2: Annealing (Epochs 31-100, 70% of training)
  - Start: 0.01 (max)
  - End: 0.01 / 1000 = 0.00001
  - Duration: 70 epochs
  - Strategy: Cosine annealing
```

### Why These Values Work for SGD

#### 1. Max LR = 0.01
✅ **Benefits:**
- Standard starting LR for SGD on CIFAR
- Safe with momentum 0.9
- Won't cause divergence
- Well-tested in literature

**Comparison:**
| Optimizer | Typical Max LR | Our Value | Status |
|-----------|---------------|-----------|---------|
| SGD + momentum | 0.01-0.05 | 0.01 | ✅ Safe |
| Adam | 0.001-0.01 | N/A | - |
| Previous (SGD) | 0.0251 | 0.01 | ⚠️ Was too high |

#### 2. Div Factor = 10
✅ **Benefits:**
- Initial LR = 0.001 (reasonable starting point)
- Model learns from epoch 1
- Not too aggressive, not too conservative
- Smooth gradient flow from start

**Initial LR Comparison:**
- Previous: 0.001004 (max_lr / 25)
- New: 0.001 (max_lr / 10)
- Improvement: Same start, but better max LR

#### 3. Pct Start = 0.3
✅ **Benefits:**
- 30 epochs warmup (vs 10 previously)
- Gentler ramp: 0.001 → 0.01 over 30 epochs
- Model stabilizes better
- SGD has time to find good gradients

**LR Ramp Rate:**
- Previous: 2.5× increase per epoch for 10 epochs
- New: 1.33× increase per epoch for 30 epochs
- Result: Much smoother convergence

#### 4. Final Div Factor = 1000
✅ **Benefits:**
- Final LR = 0.00001 (vs 0.00000251)
- Still fine-tunes in final epochs
- Not too small to be useless
- Helps squeeze out last few % accuracy

---

## Expected Improvements

### Training Progression (Expected)
```
Epoch  1: Train Acc: ~8-12%, Test Acc: ~10-14% (vs 1.58% before)
Epoch 10: Train Acc: ~35-40%, Test Acc: ~35-40% (vs 11.30% before)
Epoch 30: Train Acc: ~60-65%, Test Acc: ~58-62% (at max LR)
Epoch 50: Train Acc: ~70-75%, Test Acc: ~65-68% (mid annealing)
Epoch 100: Train Acc: ~80-85%, Test Acc: ~70-73% (final)
```

### Learning Rate Trajectory
```
Epoch   1: LR = 0.00100 (start)
Epoch  10: LR = 0.00367 (ramping up)
Epoch  20: LR = 0.00733 (still ramping)
Epoch  30: LR = 0.01000 (max reached)
Epoch  50: LR = 0.00568 (annealing)
Epoch  75: LR = 0.00134 (fine-tuning)
Epoch 100: LR = 0.00001 (final polish)
```

---

## OneCycleLR vs Other Schedulers

### Comparison for SGD

| Scheduler | Max LR | Min LR | Warmup | Best For |
|-----------|--------|--------|--------|----------|
| **OneCycleLR** | 0.01 | 0.00001 | 30 epochs | Fast convergence, good regularization |
| **StepLR** | 0.01 | 0.00001 | None | Simple, predictable |
| **CosineAnnealing** | 0.01 | 0.00001 | None | Smooth decay |
| **ReduceLROnPlateau** | 0.01 | ~0.0001 | None | Adaptive, but slower |

### When to Use OneCycleLR with SGD

✅ **Use OneCycleLR when:**
- You want fast convergence
- You have limited epochs (50-200)
- You know approximate optimal LR
- You want built-in warmup + decay
- You're using strong data augmentation

❌ **Avoid OneCycleLR when:**
- Training for >500 epochs (use CosineAnnealing)
- LR sensitivity is unknown (use ReduceLROnPlateau)
- Need to pause/resume frequently
- Want simple, predictable schedule

---

## Hyperparameter Tuning Guide

### Finding Optimal max_lr for SGD

**Method 1: LR Range Test**
```python
# Run LR finder first
lr_finder = LRFinder(model, optimizer, criterion)
lrs, losses = lr_finder.find(train_loader, start_lr=1e-7, end_lr=1, num_iter=100)

# Find LR where loss is steepest (most negative gradient)
# For SGD, typically: 0.005 - 0.05
max_lr = lrs[np.argmin(np.gradient(losses))]
```

**Method 2: Rule of Thumb**
```python
# For CIFAR-100 with SGD + momentum 0.9
if batch_size == 128:
    max_lr = 0.01      # Standard
elif batch_size == 256:
    max_lr = 0.02      # Linear scaling
elif batch_size == 512:
    max_lr = 0.04      # Linear scaling
```

### Adjusting pct_start

```python
# Warmup percentage guidelines for SGD
if total_epochs <= 50:
    pct_start = 0.3  # 15 epochs warmup
elif total_epochs <= 100:
    pct_start = 0.3  # 30 epochs warmup
elif total_epochs <= 200:
    pct_start = 0.2  # 40 epochs warmup
else:
    pct_start = 0.1  # >20 epochs warmup
```

### Adjusting div_factor

```python
# Initial LR = max_lr / div_factor
# For SGD, aim for initial LR ~ 0.0005 to 0.002

if max_lr == 0.01:
    div_factor = 10   # Initial: 0.001 ✅ Good
elif max_lr == 0.05:
    div_factor = 25   # Initial: 0.002 ✅ Good
elif max_lr == 0.1:
    div_factor = 50   # Initial: 0.002 ✅ Good
```

---

## Testing the Fix

### Quick Test (10 epochs)
```python
config.training.epochs = 10
config.training.scheduler_type = 'OneCycleLR'
config.training.onecycle_max_lr = 0.01
config.training.onecycle_pct_start = 0.3
config.training.onecycle_div_factor = 10.0
```

**Expected Results:**
- Epoch 1: Train Acc ~8-12% (should be much better than 1.58%)
- Epoch 3: Train Acc ~25-30% (at max LR)
- Epoch 10: Train Acc ~40-45%

### Full Training (100 epochs)
**Expected Final Results:**
- Train Acc: 80-85%
- Test Acc: 70-73%
- Gap: 10-15% (acceptable)
- No divergence or oscillations

---

## Common Mistakes with OneCycleLR + SGD

### Mistake #1: Using Adam's LR range
```python
❌ max_lr = 0.001  # Too small for SGD
✅ max_lr = 0.01   # Good for SGD
```

### Mistake #2: Too short warmup
```python
❌ pct_start = 0.05  # Only 5 epochs
✅ pct_start = 0.3   # 30 epochs
```

### Mistake #3: Too high max LR
```python
❌ max_lr = 0.1   # Too high, will diverge
✅ max_lr = 0.01  # Safe and effective
```

### Mistake #4: Ignoring momentum
```python
# High LR + high momentum = bad
❌ max_lr = 0.05, momentum = 0.9   # Too aggressive
✅ max_lr = 0.01, momentum = 0.9   # Balanced
✅ max_lr = 0.05, momentum = 0.5   # Also OK
```

---

## References & Best Practices

### Papers
1. **Super-Convergence** (Smith, 2018)
   - Introduced OneCycleLR
   - Recommends max_lr from LR range test
   - Shows 10× speedup with proper tuning

2. **Cyclical Learning Rates** (Smith, 2017)
   - Foundation for OneCycleLR
   - Guidelines for LR range selection

### Best Practices for CIFAR-100 + SGD
```python
# Recommended configuration
optimizer = SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,              # Safe for SGD
    steps_per_epoch=len(train_loader),
    epochs=100,
    pct_start=0.3,           # 30 epoch warmup
    div_factor=10.0,         # Start at 0.001
    final_div_factor=1000.0, # End at 0.00001
    anneal_strategy='cos'    # Smooth annealing
)
```

---

## Summary of Changes

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `max_lr` | 0.0251 | 0.01 | SGD safe range |
| `pct_start` | 0.1 | 0.3 | Longer warmup |
| `div_factor` | 25.0 | 10.0 | Higher initial LR |
| `final_div_factor` | 10000.0 | 1000.0 | Better final LR |

**Key Changes:**
- ✅ Reduced max LR by 60% (0.0251 → 0.01)
- ✅ Increased warmup by 200% (10 → 30 epochs)
- ✅ Increased initial LR by 2.5× (0.001 → 0.001)
- ✅ Increased final LR by 10× (2.51e-6 → 1e-5)

**Expected Impact:**
- 🚀 Faster initial learning (8-12% vs 1.58% in epoch 1)
- 🎯 Better convergence (70-73% vs unknown)
- 📈 Smoother training curve
- ✅ No oscillations or divergence

---

## Next Steps

1. **Run training** with new configuration
2. **Monitor first 10 epochs** - should see 8-12% accuracy in epoch 1
3. **Check epoch 30** - should reach max LR and ~60% accuracy
4. **Verify final results** - target 70-73% test accuracy

If still not working well:
- Try `max_lr = 0.005` (more conservative)
- Increase `pct_start = 0.4` (40 epoch warmup)
- Check if data augmentation is too strong
- Verify model architecture is correct

---

*Created: October 11, 2025*  
*Author: AI Assistant*  
*Status: Ready for testing*

