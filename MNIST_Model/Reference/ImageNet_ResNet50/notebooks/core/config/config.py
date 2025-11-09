"""
Model Configuration for ResNet50
Contains configuration classes for model architecture and training.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    """Configuration for ResNet50 model architecture."""
    input_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 1000
    dropout_rate: float = 0.0
    layers: Tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50 layer structure
    use_pretrained: bool = False
    pretrained_path: Optional[str] = None
    model_name: str = "resnet50"
