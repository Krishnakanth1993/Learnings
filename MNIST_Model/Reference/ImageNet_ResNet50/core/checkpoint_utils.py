"""
Utility functions for safely loading checkpoints with PyTorch 2.6+
"""
import torch
from .config import TrainingConfig


def load_checkpoint_safely(checkpoint_path: str, device: torch.device = None):
    """
    Safely load a checkpoint with proper handling of TrainingConfig serialization.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Optional device to map tensors to
    
    Returns:
        The loaded checkpoint dictionary
    """
    try:
        # First try loading with weights_only=True and TrainingConfig in safe globals
        try:
            with torch.serialization.safe_globals([TrainingConfig]):
                return torch.load(checkpoint_path, map_location=device, weights_only=True)
        except AttributeError:
            # Fallback for older PyTorch versions that don't have context manager
            torch.serialization.add_safe_globals([TrainingConfig])
            return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        # As a last resort, load with weights_only=False since we trust our own checkpoint
        print(f"Warning: Falling back to unsafe loading (weights_only=False): {str(e)}")
        return torch.load(checkpoint_path, map_location=device, weights_only=False)