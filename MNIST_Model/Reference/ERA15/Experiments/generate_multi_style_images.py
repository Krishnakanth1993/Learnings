"""
Multi-Style Textual Inversion Image Generation

This script demonstrates how to generate images using 5 different learned style embeddings 
with Stable Diffusion. Each style will be applied to the same base prompt with different 
random seeds.
"""

import torch
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


def generate_with_style(
    style_name,
    style_file,
    base_prompt,
    seed,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512
):
    """
    Generate an image using a specific style embedding.
    
    Args:
        style_name: Name of the style (for display)
        style_file: Path to the .bin file containing the learned embedding
        base_prompt: Text prompt (should include a placeholder for the style token)
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        height: Image height
        width: Image width
    
    Returns:
        PIL Image
    """
    print(f"\n{'='*60}")
    print(f"Generating image with style: {style_name}")
    print(f"{'='*60}")
    
    # Set random seed - create generator on the correct device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load the learned embedding
    learned_embeds_dict = torch.load(style_file, map_location=device)
    
    # Extract the token string and embedding vector
    # The .bin file contains a dictionary with {token_string: embedding_tensor}
    style_token = list(learned_embeds_dict.keys())[0]
    style_embedding = learned_embeds_dict[style_token].to(device)
    
    print(f"Style token: {style_token}")
    print(f"Original embedding shape: {style_embedding.shape}")
    
    # Get expected embedding dimension from text encoder
    expected_dim = text_encoder.get_input_embeddings().weight.shape[1]
    
    # Handle dimension mismatch (some embeddings are 1024-dim, model expects 768-dim)
    if style_embedding.shape[0] != expected_dim:
        print(f"Warning: Embedding dimension {style_embedding.shape[0]} doesn't match expected {expected_dim}")
        if style_embedding.shape[0] == 1024 and expected_dim == 768:
            # Project 1024-dim to 768-dim using a simple linear projection
            # We'll use the first 768 dimensions as a simple approach
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
    
    # IMPORTANT: Directly set the learned embedding in the text encoder's embedding layer
    # This is the correct way to inject custom embeddings
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
    
    # Get text embeddings using the standard forward pass
    # The learned embedding is already in the embedding layer
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
    # Base prompt - use <style> as placeholder for the style token
    base_prompt = "A mouse in the style of <style>"
    
    # Generate images for each style
    generated_images = {}
    seeds = [42, 123, 456, 789, 1024]  # Different seed for each style
    
    for (style_name, filename), seed in zip(styles.items(), seeds):
        style_file = base_dir / filename
        
        try:
            image = generate_with_style(
                style_name=style_name,
                style_file=style_file,
                base_prompt=base_prompt,
                seed=seed,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            generated_images[style_name] = image
            
            # Save the image
            output_path = base_dir / f"output_{style_name}_seed{seed}.png"
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
            axes[idx].set_title(f"{style_name.replace('_', ' ').title()}\n(seed: {seeds[idx]})", 
                              fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f"{style_name}\nFailed", 
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
    
    # Hide the last subplot if we have an odd number of images
    if len(generated_images) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle(f'Multi-Style Image Generation\nPrompt: "{base_prompt}"', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(base_dir / "all_styles_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("All images generated and saved!")
    print(f"{'='*60}")
