"""
Ice Crystal Pattern Guidance for Stable Diffusion

This module provides a custom loss function to guide image generation towards 
transparent ice crystal patterns with sharp edges and crystalline structures.
"""

import torch
import torch.nn.functional as F


def ice_crystal_loss(images):
    """
    Calculate loss to encourage ice crystal patterns with transparency and sharp edges.
    
    Ice crystal characteristics:
    1. High frequency details (sharp edges, crystalline structures)
    2. Transparency (lighter colors, high brightness in certain areas)
    3. Contrast between transparent and opaque regions
    4. Geometric patterns (hexagonal/angular structures)
    
    Args:
        images: Tensor of shape (batch, 3, height, width) in range [0, 1]
    
    Returns:
        Scalar loss value (lower = more ice crystal-like)
    """
    
    # 1. Edge Detection Loss - Encourage sharp, high-frequency details
    # Use Sobel filters to detect edges
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    
    # Apply to each channel
    edges_x = F.conv2d(images, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    edges_y = F.conv2d(images, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
    
    # We want HIGH edge magnitude (sharp crystalline structures)
    # So we minimize the negative of edge magnitude
    edge_loss = -edge_magnitude.mean()
    
    
    # 2. Brightness/Transparency Loss - Encourage lighter, transparent-looking regions
    # Ice crystals are often transparent/translucent with bright highlights
    brightness = images.mean(dim=1, keepdim=True)  # Average across RGB channels
    
    # Encourage high brightness (transparency effect)
    brightness_loss = -brightness.mean()
    
    
    # 3. Contrast Loss - Ice crystals have high local contrast
    # Calculate local standard deviation to encourage variation
    kernel_size = 5
    local_mean = F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size//2)
    local_variance = F.avg_pool2d((images - local_mean)**2, kernel_size, stride=1, padding=kernel_size//2)
    local_std = torch.sqrt(local_variance + 1e-8)
    
    # We want HIGH local contrast
    contrast_loss = -local_std.mean()
    
    
    # 4. Color Loss - Ice crystals often have cool tones (blues, whites, cyans)
    # Encourage blue/cyan channels over red
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    
    # Penalize red, reward blue and green (for cyan/blue tones)
    cool_tone_loss = r.mean() - (b.mean() + g.mean()) / 2
    
    
    # 5. High Frequency Loss - Ice crystals have intricate, high-frequency patterns
    # Use Laplacian to detect high-frequency content
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                    dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    high_freq = F.conv2d(images, laplacian_kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
    
    # We want HIGH high-frequency content
    high_freq_loss = -torch.abs(high_freq).mean()
    
    
    # Combine all losses with weights
    total_loss = (
        2.0 * edge_loss +           # Sharp edges (most important)
        1.5 * brightness_loss +      # Transparency/brightness
        1.0 * contrast_loss +        # Local contrast
        0.5 * cool_tone_loss +       # Cool color tones
        1.5 * high_freq_loss         # High-frequency details
    )
    
    return total_loss


def ice_crystal_loss_simple(images):
    """
    Simplified version focusing on the most important ice crystal characteristics.
    
    Args:
        images: Tensor of shape (batch, 3, height, width) in range [0, 1]
    
    Returns:
        Scalar loss value
    """
    # 1. Edge sharpness
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
    
    edges_x = F.conv2d(images, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    edges_y = F.conv2d(images, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
    
    # 2. Brightness (transparency)
    brightness = images.mean()
    
    # Combine: maximize edges and brightness
    loss = -(edge_magnitude.mean() + brightness)
    
    return loss


# Example usage code
if __name__ == "__main__":
    # Test the loss function
    import matplotlib.pyplot as plt
    
    # Create a test image
    test_image = torch.rand(1, 3, 512, 512)
    
    loss = ice_crystal_loss(test_image)
    print(f"Ice crystal loss: {loss.item()}")
    
    loss_simple = ice_crystal_loss_simple(test_image)
    print(f"Ice crystal loss (simple): {loss_simple.item()}")
