"""
CIFAR100 ResNet-34 Model Definition with Bottleneck Layers
Contains the model architecture classes for CIFAR100 classification.

This module provides:
- ModelConfig: Configuration for model architecture
- BottleneckBlock: 1x1 bottleneck convolution block
- BasicBlock: Basic residual block
- CIFAR100ResNet34: ResNet-34 architecture for CIFAR-100

Author: Krishnakanth
Date: 2025-10-10
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from io import StringIO
from typing import Tuple, Dict, Any
from dataclasses import dataclass


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_channels: int = 3
    input_size: Tuple[int, int] = (32, 32)
    num_classes: int = 100
    dropout_rate: float = 0.05


# =============================================================================
# BOTTLENECK BLOCK WITH 1x1 CONVOLUTIONS
# =============================================================================

class BottleneckBlock(nn.Module):
    """
    1x1 Bottleneck block for ResNet architecture.
    Reduces computational complexity by using 1x1 convolutions to reduce and expand channels.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        super(BottleneckBlock, self).__init__()
        
        # First 1x1 conv: reduces channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # 3x3 conv: main convolution
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # Second 1x1 conv: expands channels back
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


# =============================================================================
# RESNET-34 FOR CIFAR-100
# =============================================================================

class CIFAR100ResNet34(nn.Module):
    """
    ResNet-34 architecture for CIFAR-100.
    Uses BasicBlock with the 3-4-6-3 layer structure of ResNet-34.
    """
    
    def __init__(self, config: ModelConfig):
        super(CIFAR100ResNet34, self).__init__()
        self.config = config
        
        # For CIFAR-32x32, use modified initial layer (no stride/pooling)
        self.conv1 = nn.Conv2d(config.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet-34 uses BasicBlock with [3, 4, 6, 3] blocks per layer
        # Layer 1: 64 channels, 3 blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 3, stride=1, dropout_rate=config.dropout_rate)
        
        # Layer 2: 128 channels, 4 blocks
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 4, stride=2, dropout_rate=config.dropout_rate)
        
        # Layer 3: 256 channels, 6 blocks
        self.layer3 = self._make_layer(BasicBlock, 128, 256, 6, stride=2, dropout_rate=config.dropout_rate)
        
        # Layer 4: 512 channels, 3 blocks
        self.layer4 = self._make_layer(BasicBlock, 256, 512, 3, stride=2, dropout_rate=config.dropout_rate)
        
        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(512, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, dropout_rate=0.0):
        """Create a layer with specified number of blocks."""
        downsample = None
        
        # Downsample for first block in layer if needed
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # First block with potential downsampling
        layers.append(block(in_channels, out_channels, stride, downsample, dropout_rate))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet layers (3-4-6-3 blocks)
        x = self.layer1(x)  # 3 blocks
        x = self.layer2(x)  # 4 blocks
        x = self.layer3(x)  # 6 blocks
        x = self.layer4(x)  # 3 blocks
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_model_summary(self, input_size: Tuple[int, int, int], logger) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        logger.info("Generating ResNet-34 summary...")
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        summary(self, input_size)
        sys.stdout = old_stdout
        
        summary_text = captured_output.getvalue()
        
        # Parse the summary text to find the total parameters
        total_params_line = [line for line in summary_text.splitlines() 
                           if "Total params:" in line]
        if total_params_line:
            total_params_str = total_params_line[0].split(":")[1].strip().replace(",", "")
            total_params = int(total_params_str)
        else:
            # Fallback to manual calculation
            total_params = sum(p.numel() for p in self.parameters())
        
        # Check for specific layer types
        has_batchnorm = any(isinstance(module, nn.BatchNorm2d) for module in self.modules())
        has_dropout = any(isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d) 
                         for module in self.modules())
        has_fc = any(isinstance(module, nn.Linear) for module in self.modules())
        has_gap = any(isinstance(module, nn.AvgPool2d) or isinstance(module, nn.AdaptiveAvgPool2d) 
                     for module in self.modules())
        
        model_info = {
            'total_params': total_params,
            'has_batchnorm': has_batchnorm,
            'has_dropout': has_dropout,
            'has_fc_or_gap': has_fc or has_gap,
            'summary_text': summary_text
        }
        
        # Log model information
        logger.info("ResNet-34 Architecture Summary:")
        logger.info(f"  - Total Parameters: {total_params:,}")
        logger.info(f"  - Batch Normalization: {'Yes' if has_batchnorm else 'No'}")
        logger.info(f"  - Dropout: {'Yes' if has_dropout else 'No'}")
        logger.info(f"  - FC Layers: {'Yes' if has_fc else 'No'}")
        logger.info(f"  - GAP Layers: {'Yes' if has_gap else 'No'}")
        
        # Log the detailed model summary
        logger.log_detailed_model_summary(summary_text)
        
        return model_info


# =============================================================================
# MODEL BUILDER (Builder Pattern)
# =============================================================================

class ModelBuilder:
    """Builder class for constructing neural network models."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder state."""
        self._model = None
        return self
    
    def build_resnet34(self, config: ModelConfig) -> 'ModelBuilder':
        """Build ResNet-34 for CIFAR-100."""
        self._model = CIFAR100ResNet34(config)
        return self
    
    def build_cifar100_model(self, config: ModelConfig) -> 'ModelBuilder':
        """Build CIFAR-100 ResNet-34 model. Alias for build_resnet34."""
        self._model = CIFAR100ResNet34(config)
        return self
    
    def build(self) -> 'CIFAR100ResNet34':
        """Return the built model."""
        if self._model is None:
            raise ValueError("No model has been built yet. Call a build method first.")
        return self._model


# Aliases for compatibility
CIFAR100Model = CIFAR100ResNet34
CIFAR100ResNet18 = CIFAR100ResNet34  # Backward compatibility alias