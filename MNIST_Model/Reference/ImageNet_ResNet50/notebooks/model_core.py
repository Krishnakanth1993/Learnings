"""
ImageNet-1K ResNet50 Model Loading Functions
Contains functions for loading and using the ResNet50 model from the core module.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import Tuple, Optional, Dict, Any

# Import from core module
from core import ResNet50, ModelConfig


def load_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[ResNet50, torch.device]:
    """
    Load the trained ResNet50 model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model on (CPU/CUDA)
        
    Returns:
        Tuple of (model, device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model configuration
    config = ModelConfig(
        input_channels=3,
        input_size=(224, 224),
        num_classes=1000,
        dropout_rate=0.0,
        layers=(3, 4, 6, 3),  # ResNet-50 layer structure
        use_pretrained=False,
        pretrained_path=None,
        model_name="resnet50"
    )
    
    # Initialize model
    model = ResNet50(config)
    
    # Load trained weights
    try:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle any mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"Model loaded successfully!")
        if missing_keys:
            print(f"Missing keys (ignored): {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)}")
        
        model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded on {device}")
        print(f"Total parameters: {total_params:,}")
        
        return model, device
        
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


def get_transform():
    """
    Get image transformation pipeline for ImageNet preprocessing.
    
    Returns:
        torchvision.transforms.Compose: Image preprocessing pipeline
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL image for model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Add batch dimension


def get_model_info(model: ResNet50) -> Dict[str, Any]:
    """
    Get information about the model.
    
    Args:
        model: ResNet50 model instance
        
    Returns:
        Dict containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_name': model.config.model_name,
        'num_classes': model.config.num_classes,
        'input_size': model.config.input_size
    }
