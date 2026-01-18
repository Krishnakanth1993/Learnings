# Memory Optimization Guide for Ice Crystal Guidance

## Problem: CUDA Out of Memory Error

Ice crystal guidance requires additional GPU memory because it:
1. Decodes latents to image space during generation
2. Computes gradients for the loss function
3. Stores intermediate tensors

## Solutions Applied

### 1. **Automatic Memory Cleanup**
The script now automatically:
- Clears CUDA cache before and after guidance steps
- Deletes intermediate tensors immediately after use
- Uses `torch.cuda.empty_cache()` to free memory

### 2. **Reduced Guidance Frequency**
Default changed from every 5 steps to every 10 steps:
```python
GUIDANCE_FREQUENCY = 10  # Less frequent = less memory usage
```

### 3. **Disabled Autocast**
Uses `torch.cuda.amp.autocast(enabled=False)` for stability during VAE decoding.

---

## If You Still Get Out of Memory Errors

### Option 1: Increase Guidance Frequency (Easiest)
```python
GUIDANCE_FREQUENCY = 15  # or even 20
```
- Less frequent guidance = less memory usage
- Still effective, just slightly weaker effect

### Option 2: Reduce Image Resolution
```python
# In the generate_with_style call:
height=384,  # Instead of 512
width=384,
```

### Option 3: Process One Style at a Time
Comment out styles you don't need:
```python
styles = {
    "8bit": "8bit_learned_embeds.bin",
    # "ahx_beta": "ahx_beta_learned_embeds.bin",  # Skip this one
    # "dr_strange": "dr_strangelearned_embeds.bin",
    # etc...
}
```

### Option 4: Reduce Inference Steps
```python
num_inference_steps=30,  # Instead of 50
```
- Faster generation
- Uses less memory
- Slightly lower quality

### Option 5: Use CPU for VAE Decoding (Slowest but works)
Add this modification to the script:

```python
# Before VAE decode in ice crystal guidance section:
vae_cpu = vae.cpu()
denoised_images = vae_cpu.decode((1 / 0.18215) * latents_x0.cpu()).sample / 2 + 0.5
denoised_images = denoised_images.to(device)
vae = vae.to(device)
```

### Option 6: Disable Ice Crystal Guidance Temporarily
```python
USE_ICE_CRYSTAL_GUIDANCE = False
```
Generate normal images first, then enable for specific styles.

---

## Memory Usage Comparison

| Configuration | Approx. VRAM | Speed |
|--------------|--------------|-------|
| No guidance | ~4-6 GB | Fast |
| Guidance every 10 steps | ~6-8 GB | Medium |
| Guidance every 5 steps | ~8-10 GB | Slower |
| Guidance every 3 steps | ~10-12 GB | Slowest |

---

## Recommended Settings by GPU

### 6 GB VRAM (e.g., RTX 2060)
```python
USE_ICE_CRYSTAL_GUIDANCE = True
GUIDANCE_FREQUENCY = 15
num_inference_steps = 30
# Process 1-2 styles at a time
```

### 8 GB VRAM (e.g., RTX 3070)
```python
USE_ICE_CRYSTAL_GUIDANCE = True
GUIDANCE_FREQUENCY = 10
num_inference_steps = 50
# Can process all 5 styles
```

### 10+ GB VRAM (e.g., RTX 3080, 4090)
```python
USE_ICE_CRYSTAL_GUIDANCE = True
GUIDANCE_FREQUENCY = 5
num_inference_steps = 50
# No issues
```

---

## Quick Fix Commands

If you get OOM error, try these in order:

1. **Increase frequency:**
   ```python
   GUIDANCE_FREQUENCY = 15
   ```

2. **Reduce steps:**
   ```python
   num_inference_steps = 30
   ```

3. **Process one at a time:**
   ```python
   # Comment out 4 styles, keep only 1
   ```

4. **Disable guidance:**
   ```python
   USE_ICE_CRYSTAL_GUIDANCE = False
   ```

---

## Monitoring Memory Usage

Add this at the start of your script to monitor VRAM:
```python
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
# After each generation:
if torch.cuda.is_available():
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"VRAM cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

The script has been optimized with automatic memory cleanup. Try running it again with the new `GUIDANCE_FREQUENCY = 10` setting!
