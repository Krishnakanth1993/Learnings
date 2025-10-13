"""
ResNet-50 Model Architecture for ImageNet
Implements ResNet-50 with Bottleneck blocks following OOP principles.

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from typing import Tuple, Dict, Any, Optional, List
from io import StringIO
import sys

from .config import ModelConfig


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152.
    Structure: 1x1 conv (reduce) -> 3x3 conv -> 1x1 conv (expand)
    Expansion factor: 4x
    """
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, dropout_rate: float = 0.0):
        super(BottleneckBlock, self).__init__()
        
        # 1x1 conv: reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv: main feature extraction
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv: expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class ResNet50(nn.Module):
    """
    ResNet-50 architecture for ImageNet classification.
    
    Structure:
    - Initial conv: 7x7, stride=2
    - MaxPool: 3x3, stride=2
    - Layer1: 3 bottleneck blocks (64 -> 256 channels)
    - Layer2: 4 bottleneck blocks (128 -> 512 channels)
    - Layer3: 6 bottleneck blocks (256 -> 1024 channels)
    - Layer4: 3 bottleneck blocks (512 -> 2048 channels)
    - Global Average Pooling
    - FC: 2048 -> num_classes
    
    Total Parameters: ~25.6M
    """
    
    def __init__(self, config: ModelConfig):
        super(ResNet50, self).__init__()
        self.config = config
        self.in_channels = 64
        
        # Initial convolutional layer (7x7, stride=2 for ImageNet)
        self.conv1 = nn.Conv2d(config.input_channels, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-50 layers: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(BottleneckBlock, 64, config.layers[0], 
                                        stride=1, dropout_rate=config.dropout_rate)
        self.layer2 = self._make_layer(BottleneckBlock, 128, config.layers[1], 
                                        stride=2, dropout_rate=config.dropout_rate)
        self.layer3 = self._make_layer(BottleneckBlock, 256, config.layers[2], 
                                        stride=2, dropout_rate=config.dropout_rate)
        self.layer4 = self._make_layer(BottleneckBlock, 512, config.layers[3], 
                                        stride=2, dropout_rate=config.dropout_rate)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if specified
        if config.use_pretrained and config.pretrained_path:
            self._load_pretrained(config.pretrained_path)
    
    def _make_layer(self, block: type, out_channels: int, num_blocks: int,
                    stride: int = 1, dropout_rate: float = 0.0) -> nn.Sequential:
        """Create a layer with specified number of bottleneck blocks."""
        downsample = None
        
        # Downsample if stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # First block (with potential downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet-50 bottleneck layers
        x = self.layer1(x)  # 3 blocks
        x = self.layer2(x)  # 4 blocks
        x = self.layer3(x)  # 6 blocks
        x = self.layer4(x)  # 3 blocks
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'layer4') -> torch.Tensor:
        """
        Extract feature maps from a specific layer (for Grad-CAM).
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from
        
        Returns:
            Feature maps from specified layer
        """
        # Forward pass up to specified layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if layer_name == 'layer1':
            return x
        
        x = self.layer2(x)
        if layer_name == 'layer2':
            return x
        
        x = self.layer3(x)
        if layer_name == 'layer3':
            return x
        
        x = self.layer4(x)
        if layer_name == 'layer4':
            return x
        
        return x
    
    def get_model_summary(self, input_size: Tuple[int, int, int], logger) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        logger.info(f"Generating {self.config.model_name.upper()} summary...")
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            summary(self, input_size)
        except:
            print("Could not generate torchsummary output")
        sys.stdout = old_stdout
        
        summary_text = captured_output.getvalue()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Check layer types
        has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in self.modules())
        has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in self.modules())
        has_fc = any(isinstance(m, nn.Linear) for m in self.modules())
        has_gap = any(isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)) for m in self.modules())
        
        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'has_batchnorm': has_batchnorm,
            'has_dropout': has_dropout,
            'has_fc': has_fc,
            'has_gap': has_gap,
            'summary_text': summary_text
        }
        
        # Log model information
        logger.info(f"{self.config.model_name.upper()} Architecture Summary:")
        logger.info(f"  Total Parameters: {total_params:,}")
        logger.info(f"  Trainable Parameters: {trainable_params:,}")
        logger.info(f"  Model Size: {total_params * 4 / (1024**2):.2f} MB")
        logger.info(f"  Batch Normalization: {'Yes' if has_batchnorm else 'No'}")
        logger.info(f"  Dropout: {'Yes' if has_dropout else 'No'}")
        logger.info(f"  FC Layers: {'Yes' if has_fc else 'No'}")
        logger.info(f"  Global Pooling: {'Yes' if has_gap else 'No'}")
        
        if summary_text:
            logger.info("=" * 70)
            logger.info("DETAILED MODEL ARCHITECTURE")
            logger.info("=" * 70)
            for line in summary_text.splitlines():
                logger.info(line)
            logger.info("=" * 70)
        
        return model_info


class ModelBuilder:
    """Builder class for constructing ResNet models."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self._model = None
        return self
    
    def build_resnet50(self, config: ModelConfig) -> 'ModelBuilder':
        """Build ResNet-50 model."""
        self._model = ResNet50(config)
        return self
    
    def build(self) -> ResNet50:
        """Return the built model."""
        if self._model is None:
            raise ValueError("No model has been built. Call build_resnet50() first.")
        return self._model
    
    def to_device(self, device: torch.device) -> 'ModelBuilder':
        """Move model to specified device."""
        if self._model is not None:
            self._model = self._model.to(device)
        return self


# Convenience function
def create_resnet50(config: ModelConfig, device: Optional[torch.device] = None) -> ResNet50:
    """
    Convenience function to create ResNet-50 model.
    
    Args:
        config: Model configuration
        device: Target device (CPU/CUDA)
    
    Returns:
        ResNet-50 model
    """
    model = ResNet50(config)
    if device is not None:
        model = model.to(device)
    return model

