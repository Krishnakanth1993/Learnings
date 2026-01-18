# Ice Crystal Pattern Guidance for Stable Diffusion

## Overview

This guide explains how to replace the `blue_loss` with a custom **ice crystal pattern loss** that guides Stable Diffusion to generate images with transparent, crystalline structures.

---

## What is Ice Crystal Loss?

The ice crystal loss function encourages the following visual characteristics:

1. **Sharp Edges** - Crystalline structures have well-defined, angular edges
2. **Transparency** - Ice crystals are translucent with bright highlights
3. **High Contrast** - Strong local contrast between transparent and opaque regions
4. **Cool Tones** - Blues, cyans, and whites (characteristic of ice)
5. **High-Frequency Details** - Intricate, fine-grained patterns

---

## Files Created

### 1. `ice_crystal_guidance.py`
Contains two loss functions:

- **`ice_crystal_loss(images)`** - Full version with all 5 characteristics
- **`ice_crystal_loss_simple(images)`** - Simplified version (faster, focuses on edges + brightness)

### 2. `generate_with_ice_crystal_guidance.py`
Complete example showing how to integrate the loss into your generation loop.

---

## How to Use in Your Notebook

### Option 1: Copy-Paste the Loss Function

Add this to your notebook cell:

```python
import torch
import torch.nn.functional as F

def ice_crystal_loss(images):
    """Encourage ice crystal patterns with sharp edges and transparency."""
    
    # 1. Edge Detection - Sharp crystalline structures
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    
    edges_x = F.conv2d(images, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    edges_y = F.conv2d(images, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
    edge_loss = -edge_magnitude.mean()
    
    # 2. Brightness - Transparency effect
    brightness_loss = -images.mean()
    
    # 3. Local Contrast
    kernel_size = 5
    local_mean = F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size//2)
    local_variance = F.avg_pool2d((images - local_mean)**2, kernel_size, stride=1, padding=kernel_size//2)
    local_std = torch.sqrt(local_variance + 1e-8)
    contrast_loss = -local_std.mean()
    
    # 4. Cool tones (blue/cyan)
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    cool_tone_loss = r.mean() - (b.mean() + g.mean()) / 2
    
    # 5. High-frequency details
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                    dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    high_freq = F.conv2d(images, laplacian_kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
    high_freq_loss = -torch.abs(high_freq).mean()
    
    # Combine with weights
    total_loss = (
        2.0 * edge_loss +
        1.5 * brightness_loss +
        1.0 * contrast_loss +
        0.5 * cool_tone_loss +
        1.5 * high_freq_loss
    )
    
    return total_loss
```

### Option 2: Import from the Module

```python
from ice_crystal_guidance import ice_crystal_loss
```

---

## Replacing blue_loss in Your Code

### Before (with blue_loss):
```python
# Calculate loss
loss = blue_loss(denoised_images) * blue_loss_scale
```

### After (with ice_crystal_loss):
```python
# Calculate ice crystal loss
ice_crystal_loss_scale = 200  # Adjust this value
loss = ice_crystal_loss(denoised_images) * ice_crystal_loss_scale
```

---

## Complete Example

Here's your modified generation loop:

```python
# Configuration
prompt = 'A campfire (oil on canvas)'
ice_crystal_loss_scale = 200  # Adjust strength of guidance

# ... (setup code remains the same) ...

# Denoising Loop
for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
    latent_model_input = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #### ICE CRYSTAL GUIDANCE ###
    if i % 5 == 0:
        latents = latents.detach().requires_grad_()
        latents_x0 = latents - sigma * noise_pred
        denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
        
        # Use ice crystal loss instead of blue_loss
        loss = ice_crystal_loss(denoised_images) * ice_crystal_loss_scale
        
        if i % 10 == 0:
            print(i, 'ice crystal loss:', loss.item())
        
        cond_grad = torch.autograd.grad(loss, latents)[0]
        latents = latents.detach() - cond_grad * sigma**2

    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

---

## Tuning Parameters

### `ice_crystal_loss_scale`
- **Lower values (50-100)**: Subtle ice crystal effect
- **Medium values (150-250)**: Balanced guidance
- **Higher values (300-500)**: Strong ice crystal patterns (may override prompt)

### Guidance Frequency
- `if i % 5 == 0:` - Apply every 5 steps (balanced)
- `if i % 3 == 0:` - More frequent (stronger effect)
- `if i % 10 == 0:` - Less frequent (subtle effect)

### Loss Component Weights
You can adjust the weights in the loss function:
```python
total_loss = (
    2.0 * edge_loss +        # Increase for sharper edges
    1.5 * brightness_loss +  # Increase for more transparency
    1.0 * contrast_loss +    # Increase for more contrast
    0.5 * cool_tone_loss +   # Increase for bluer tones
    1.5 * high_freq_loss     # Increase for finer details
)
```

---

## Tips for Best Results

1. **Start with a relevant prompt**: "ice crystals", "frozen patterns", "crystalline structure"
2. **Adjust the scale**: Start with 200 and experiment
3. **Monitor the loss**: If it's too negative, reduce the scale
4. **Combine with style embeddings**: You can use this with textual inversion!

---

## Example Prompts That Work Well

- "A frozen lake with ice crystal patterns"
- "Transparent ice crystals on a window"
- "Crystalline ice formations in winter"
- "A campfire through frosted glass" (your example)
- "Snowflake macro photography"

---

## Troubleshooting

**Problem**: Images are too bright/washed out
- **Solution**: Reduce `brightness_loss` weight or lower `ice_crystal_loss_scale`

**Problem**: Not enough ice crystal effect
- **Solution**: Increase `ice_crystal_loss_scale` or apply guidance more frequently

**Problem**: Images look noisy
- **Solution**: Reduce `high_freq_loss` weight or use the simplified version

---

Enjoy creating beautiful ice crystal patterns! ❄️✨
