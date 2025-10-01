"""
CIFAR10 Model Definition
Contains the model architecture classes for CIFAR10 classification.

This module provides:
- ModelConfig: Configuration for model architecture
- ModelBuilder: Builder pattern for constructing models
- CIFAR10Model: The main CNN model class

Author: Krishnakanth
Date: 2025-09-28
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
    input_channels: int = 1
    input_size: Tuple[int, int] = (28, 28)
    num_classes: int = 10
    dropout_rate: float = 0.05


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
    
    def build_cifar10_model(self, config: ModelConfig) -> 'CIFAR10Model':
        """Build CIFAR10 classification model."""
        self._model = CIFAR10Model(config)
        return self._model
    
    def get_model(self) -> 'CIFAR10Model':
        """Get the built model."""
        if self._model is None:
            raise ValueError("No model has been built yet. Call build_cifar10_model() first.")
        return self._model

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout): 
        super(depthwise_separable_conv, self).__init__() 
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin) 
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1) 
  
    def forward(self, x): 
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return out

# =============================================================================
# CIFAR10 MODEL DEFINITION
# =============================================================================
class CIFAR10Model(nn.Module):
    """
    CIFAR10 classification model.
    Improved architecture with proper Batch Normalization and regularization.
    """
    
    def __init__(self, config: ModelConfig):
        super(CIFAR10Model, self).__init__()
        self.config = config
        
        # INPUT BLOCK
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 32
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 32

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 32

        # TRANSITION BLOCK 1

        self.convdialated1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=9),
            nn.BatchNorm2d(16)
        ) # output_size = 16

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16

        # CONVOLUTION BLOCK 2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 16

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 16

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 16

        # TRANSITION BLOCK 2

        self.convdialated2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=5),
            nn.BatchNorm2d(32)
        ) # output_size = 8

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8

        # CONVOLUTION BLOCK 3
        
        
        self.convblock8 = nn.Sequential(
            depthwise_separable_conv(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 8

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 8

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2,dilation = 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
            #nn.Dropout2d(0.1)
        ) # output_size = 4

        # TRANSITION BLOCK 3

        self.convdialated3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=3),
            nn.BatchNorm2d(32)
        ) # output_size = 4 

        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 4

        # CONVOLUTION BLOCK 4
        
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
        ) # output_size = 4

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2,dilation = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
        ) # output_size = 4

        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2,dilation = 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
        ) # output_size = 4

        #  # Additional convolution layer instead of GAP
        # self.convblock14 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Dropout2d(self.config.dropout_rate)
        # ) # output_size = 4

        # 1x1 Convolution to reduce channels to 10 classes
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.config.dropout_rate)
        ) # output_size = 4


        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

    
        # Fully connected layers
        self.fc0 = nn.Linear(128, 100)  
        self.fc1 = nn.Linear(100, 10)


        

    def forward(self, x):
        
        x = self.convblock1(x)


        # Convolution Block 1
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convdialated1(x)
        #x = self.pool1(x)  # 32x32 -> 16x16
        
        # Convolution Block 2
        
        x = self.convblock5(x)
        #x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convdialated2(x)
        #x = self.pool2(x)  # 16x16 -> 8x8
        
        # Convolution Block 3
        
        x = self.convblock8(x)
        #x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convdialated3(x)
        #x = self.pool3(x)  # 8x8 -> 4x4
        
        # # Convolution Block 4
        x = self.convblock11(x)
        #x = self.convblock12(x)
        x = self.convblock13(x)

         # Additional convolution layer
        #x = self.convblock14(x)  # 256 channels -> 128 channels
        
        # 1x1 Convolution to get 10 classes
        x = self.conv1x1(x)  # 128 channels -> 10 channels
        
        # Final pooling to reduce spatial dimensions
        x = self.gap(x)  # 4x4 
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten while preserving batch dimension

        
        # Fully connected layers
        x = torch.relu(self.fc0(x))
        x = self.fc1(x)
        
        # No fully connected layers needed - we already have 10 outputs
        return F.log_softmax(x, dim=-1)
    
    def get_model_summary(self, input_size: Tuple[int, int, int], logger) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        logger.info("Generating model summary...")
        
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
        has_dropout = any(isinstance(module, nn.Dropout) for module in self.modules())
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
        logger.info("Model Architecture Summary:")
        logger.info(f"  - Total Parameters: {total_params:,}")
        logger.info(f"  - Batch Normalization: {'Yes' if has_batchnorm else 'No'}")
        logger.info(f"  - Dropout: {'Yes' if has_dropout else 'No'}")
        logger.info(f"  - GAP Layers: {'Yes' if has_fc else 'No'}")
        logger.info(f"  - FC Layers: {'Yes' if has_gap else 'No'}")
        
        # Log the detailed model summary
        logger.log_detailed_model_summary(summary_text)
        
        return model_info
