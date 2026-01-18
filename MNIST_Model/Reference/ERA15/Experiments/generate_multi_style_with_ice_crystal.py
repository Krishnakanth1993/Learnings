"""
Multi-Style Textual Inversion with Optional Ice Crystal Guidance

This script combines multi-style textual inversion with optional ice crystal pattern guidance.
You can generate images with different learned style embeddings and optionally apply 
ice crystal guidance to create crystalline, transparent effects.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model configuration
model_id = "CompVis/stable-diffusion-v1-4"

# Load the models
print("Loading models...")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

# Initialize scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

print("Models loaded successfully!")

# Define style embeddings
styles = {
    "8bit": "8bit_learned_embeds.bin",
    "ahx_beta": "ahx_beta_learned_embeds.bin",
    "dr_strange": "dr_strangelearned_embeds.bin",
    "max_naylor": "max_naylorlearned_embeds.bin",
    "smiling_friend": "smiling-friend-style_learned_embeds.bin"
}

# Base directory for embeddings
base_dir = Path(r"c:\Users\krish\Documents\Krishnakanth\Learnings\Learnings\MNIST_Model\Reference\ERA15")

# Verify all embedding files exist
print("\nVerifying embedding files:")
for style_name, filename in styles.items():
    filepath = base_dir / filename
    if filepath.exists():
        print(f"✓ {style_name}: {filename}")
    else:
        print(f"✗ {style_name}: {filename} NOT FOUND")


def ice_crystal_loss(images):
    """
    Calculate loss to encourage TRANSPARENT ice crystal patterns as an overlay.
    This version preserves the original content while adding crystalline effects.
    
    Args:
        images: Tensor of shape (batch, 3, height, width) in range [0, 1]
    
    Returns:
        Scalar loss value (lower = more ice crystal-like)
    """
    # 1. Edge Detection - Sharp crystalline structures (KEEP THIS STRONG)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    
    edges_x = F.conv2d(images, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    edges_y = F.conv2d(images, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
    
    # We want sharp edges but only in certain areas (not everywhere)
    # Use a threshold to encourage edges only where they're strong
    edge_threshold = 0.1
    strong_edges = torch.relu(edge_magnitude - edge_threshold)
    edge_loss = -strong_edges.mean()
    
    # 2. MODIFIED: Selective Brightness - Only encourage brightness in edge regions
    # This creates transparent crystals without washing out the whole image
    edge_mask = (edge_magnitude > edge_threshold).float()
    brightness = images.mean(dim=1, keepdim=True)
    
    # Only brighten areas with edges (crystal formations)
    selective_brightness = brightness * edge_mask
    brightness_loss = -selective_brightness.mean() * 0.3  # Much lower weight
    
    # 3. High-frequency details - Crystalline patterns
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                    dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    high_freq = F.conv2d(images, laplacian_kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
    
    # Encourage high-frequency content but not too aggressively
    high_freq_loss = -torch.abs(high_freq).mean() * 0.5
    
    # 4. MODIFIED: Subtle cool tones (don't force everything to be blue)
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    
    # Only encourage cool tones in bright areas (ice crystals)
    bright_mask = (brightness.squeeze(1) > 0.5).float()
    cool_tone_loss = (r * bright_mask).mean() - ((b * bright_mask).mean() + (g * bright_mask).mean()) / 2
    cool_tone_loss = cool_tone_loss * 0.2  # Very subtle
    
    # 5. NEW: Texture variance - Ice crystals have varied texture
    # Calculate local texture to encourage crystalline patterns
    kernel_size = 3
    local_mean = F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size//2)
    local_variance = F.avg_pool2d((images - local_mean)**2, kernel_size, stride=1, padding=kernel_size//2)
    
    # Encourage variance in edge regions (crystalline texture)
    texture_in_edges = local_variance * edge_mask.unsqueeze(1)
    texture_loss = -texture_in_edges.mean() * 0.5
    
    # Combine with BALANCED weights (preserve original content)
    total_loss = (
        3.0 * edge_loss +           # Sharp edges (most important)
        0.5 * brightness_loss +      # REDUCED: Selective brightness only
        0.8 * high_freq_loss +       # Crystalline details
        0.2 * cool_tone_loss +       # REDUCED: Subtle cool tones
        1.0 * texture_loss           # Crystalline texture
    )
    
    return total_loss


def generate_with_style(
    style_name,
    style_file,
    base_prompt,
    seed,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    use_ice_crystal_guidance=False,
    ice_crystal_loss_scale=200,
    guidance_frequency=5
):
    """
    Generate an image using a specific style embedding with optional ice crystal guidance.
    
    Args:
        style_name: Name of the style (for display)
        style_file: Path to the .bin file containing the learned embedding
        base_prompt: Text prompt (should include a placeholder for the style token)
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        height: Image height
        width: Image width
        use_ice_crystal_guidance: Whether to apply ice crystal pattern guidance
        ice_crystal_loss_scale: Scale for ice crystal loss (higher = stronger effect)
        guidance_frequency: Apply guidance every N steps (lower = more frequent)
    
    Returns:
        PIL Image
    """
    print(f"\n{'='*60}")
    print(f"Generating image with style: {style_name}")
    if use_ice_crystal_guidance:
        print(f"Ice Crystal Guidance: ENABLED (scale={ice_crystal_loss_scale})")
    print(f"{'='*60}")
    
    # Set random seed - create generator on the correct device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load the learned embedding
    learned_embeds_dict = torch.load(style_file, map_location=device)
    
    # Extract the token string and embedding vector
    style_token = list(learned_embeds_dict.keys())[0]
    style_embedding = learned_embeds_dict[style_token].to(device)
    
    print(f"Style token: {style_token}")
    print(f"Original embedding shape: {style_embedding.shape}")
    
    # Get expected embedding dimension from text encoder
    expected_dim = text_encoder.get_input_embeddings().weight.shape[1]
    
    # Handle dimension mismatch
    if style_embedding.shape[0] != expected_dim:
        print(f"Warning: Embedding dimension {style_embedding.shape[0]} doesn't match expected {expected_dim}")
        if style_embedding.shape[0] == 1024 and expected_dim == 768:
            style_embedding = style_embedding[:768]
            print(f"Projected embedding to {expected_dim} dimensions")
        else:
            raise ValueError(f"Cannot handle embedding dimension {style_embedding.shape[0]} -> {expected_dim}")
    
    print(f"Final embedding shape: {style_embedding.shape}")
    
    # Add the token to the tokenizer if not already present
    if style_token not in tokenizer.get_vocab():
        num_added = tokenizer.add_tokens([style_token])
        text_encoder.resize_token_embeddings(len(tokenizer))
        print(f"Added {num_added} new token(s) to tokenizer")
    
    # Get the token ID
    token_id = tokenizer.convert_tokens_to_ids(style_token)
    
    # Inject the learned embedding
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = style_embedding
    
    print(f"Injected learned embedding for token ID: {token_id}")
    
    # Replace the prompt placeholder with the actual style token
    prompt = base_prompt.replace("<style>", style_token)
    print(f"Prompt: {prompt}")
    
    # Tokenize the prompt
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    # Create unconditional embeddings for classifier-free guidance
    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Concatenate for classifier-free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )
    
    # Set scheduler timesteps
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    
    # Denoising loop
    print(f"Running denoising loop ({num_inference_steps} steps)...")
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
        
        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        #### OPTIONAL ICE CRYSTAL GUIDANCE ###
        if use_ice_crystal_guidance and i % guidance_frequency == 0:
            # Clear cache before guidance to free memory
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Requires grad on the latents
            latents = latents.detach().requires_grad_()
            
            # Get the predicted x0 (denoised prediction)
            sigma = scheduler.sigmas[i]
            latents_x0 = latents - sigma * noise_pred
            
            # Decode to image space (use smaller chunks if needed)
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
                denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)
            
            # Calculate ice crystal loss
            loss = ice_crystal_loss(denoised_images) * ice_crystal_loss_scale
            
            # Print loss occasionally
            if i % 10 == 0:
                print(f"  Step {i}, Ice Crystal Loss: {loss.item():.4f}")
            
            # Get gradient with respect to latents
            cond_grad = torch.autograd.grad(loss, latents)[0]
            
            # Modify the latents based on this gradient
            latents = latents.detach() - cond_grad * sigma**2
            
            # Clean up to free memory
            del denoised_images, loss, cond_grad
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents to image
    print("Decoding latents to image...")
    latents = 1 / 0.18215 * latents
    
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    # Convert to PIL Image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    print(f"✓ Image generated successfully!")
    return image


# Main execution
if __name__ == "__main__":
    # Configuration
    base_prompt = "A mouse in the style of <style>"
    
    # Ice crystal guidance settings
    USE_ICE_CRYSTAL_GUIDANCE = True   # Set to True to enable ice crystal guidance
    ICE_CRYSTAL_LOSS_SCALE = 50       # REDUCED: 30-80 for transparent overlay (was 100-200)
    GUIDANCE_FREQUENCY = 10            # Apply every N steps (10-15 recommended)
    
    # Generate images for each style
    generated_images = {}
    seeds = [42, 123, 456, 789, 1024]  # Different seed for each style
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Prompt: {base_prompt}")
    print(f"  Ice Crystal Guidance: {'ENABLED' if USE_ICE_CRYSTAL_GUIDANCE else 'DISABLED'}")
    if USE_ICE_CRYSTAL_GUIDANCE:
        print(f"  Ice Crystal Scale: {ICE_CRYSTAL_LOSS_SCALE}")
        print(f"  Guidance Frequency: every {GUIDANCE_FREQUENCY} steps")
    print(f"{'='*60}\n")
    
    for (style_name, filename), seed in zip(styles.items(), seeds):
        style_file = base_dir / filename
        
        try:
            image = generate_with_style(
                style_name=style_name,
                style_file=style_file,
                base_prompt=base_prompt,
                seed=seed,
                num_inference_steps=50,
                guidance_scale=7.5,
                use_ice_crystal_guidance=USE_ICE_CRYSTAL_GUIDANCE,
                ice_crystal_loss_scale=ICE_CRYSTAL_LOSS_SCALE,
                guidance_frequency=GUIDANCE_FREQUENCY
            )
            generated_images[style_name] = image
            
            # Save the image
            suffix = "_ice_crystal" if USE_ICE_CRYSTAL_GUIDANCE else ""
            output_path = base_dir / f"output_{style_name}_seed{seed}{suffix}.png"
            image.save(output_path)
            print(f"Saved to: {output_path}\n")
            
        except Exception as e:
            print(f"Error generating image for {style_name}: {e}\n")
            generated_images[style_name] = None
    
    # Display all generated images
    print("\nCreating comparison grid...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (style_name, image) in enumerate(generated_images.items()):
        if image is not None:
            axes[idx].imshow(image)
            title = f"{style_name.replace('_', ' ').title()}\n(seed: {seeds[idx]})"
            if USE_ICE_CRYSTAL_GUIDANCE:
                title += "\n❄️ Ice Crystal"
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"{style_name}\nFailed", 
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
    
    # Hide the last subplot if we have an odd number of images
    if len(generated_images) < len(axes):
        axes[-1].axis('off')
    
    title_text = f'Multi-Style Image Generation\nPrompt: "{base_prompt}"'
    if USE_ICE_CRYSTAL_GUIDANCE:
        title_text += f'\n❄️ With Ice Crystal Guidance (scale={ICE_CRYSTAL_LOSS_SCALE})'
    
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    suffix = "_ice_crystal" if USE_ICE_CRYSTAL_GUIDANCE else ""
    plt.savefig(base_dir / f"all_styles_comparison{suffix}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("All images generated and saved!")
    print(f"{'='*60}")
