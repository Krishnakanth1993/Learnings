"""
Stable Diffusion with Ice Crystal Pattern Guidance

This script demonstrates how to use custom ice crystal loss guidance
during the diffusion process to generate images with crystalline patterns.
"""

import torch
from tqdm import tqdm
from ice_crystal_guidance import ice_crystal_loss, ice_crystal_loss_simple

# Configuration
prompt = 'A campfire (oil on canvas)'  # Your base prompt
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 8
generator = torch.Generator(device=torch_device).manual_seed(32)
batch_size = 1

# Ice crystal guidance parameters
ice_crystal_loss_scale = 200  # Adjust this to control the strength of ice crystal guidance
use_simple_loss = False  # Set to True to use the simpler, faster version

# Select which loss function to use
crystal_loss_fn = ice_crystal_loss_simple if use_simple_loss else ice_crystal_loss

# Prep text embeddings
text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# Unconditional embeddings
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, 
                         return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Prep Scheduler
scheduler.set_timesteps(num_inference_steps)

# Prep latents
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device
)
latents = latents * scheduler.init_noise_sigma

# Denoising Loop with Ice Crystal Guidance
print("Starting generation with ice crystal guidance...")
for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
    # Expand the latents for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    sigma = scheduler.sigmas[i]
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    # Perform classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #### ICE CRYSTAL PATTERN GUIDANCE ###
    if i % 5 == 0:  # Apply guidance every 5 steps
        # Requires grad on the latents
        latents = latents.detach().requires_grad_()

        # Get the predicted x0 (denoised prediction)
        latents_x0 = latents - sigma * noise_pred

        # Decode to image space
        denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5  # range (0, 1)

        # Calculate ice crystal loss
        loss = crystal_loss_fn(denoised_images) * ice_crystal_loss_scale

        # Print loss occasionally
        if i % 10 == 0:
            print(f"Step {i}, Ice Crystal Loss: {loss.item():.4f}")

        # Get gradient with respect to latents
        cond_grad = torch.autograd.grad(loss, latents)[0]

        # Modify the latents based on this gradient
        # This pushes the generation towards ice crystal patterns
        latents = latents.detach() - cond_grad * sigma**2

    # Step with scheduler
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Convert final latents to image
print("Decoding final image...")
final_image = latents_to_pil(latents)[0]
final_image.save("ice_crystal_guided_output.png")
print("Image saved as 'ice_crystal_guided_output.png'")
final_image
