"""
MNIST Model Definition
Contains the model architecture classes for MNIST digit classification.

This module provides:
- ModelConfig: Configuration for model architecture
- ModelBuilder: Builder pattern for constructing models
- MNISTModel: The main CNN model class

Author: AI Assistant
Date: 2024
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
    
    def build_mnist_model(self, config: ModelConfig) -> 'MNISTModel':
        """Build MNIST classification model."""
        self._model = MNISTModel(config)
        return self._model
    
    def get_model(self) -> 'MNISTModel':
        """Get the built model."""
        if self._model is None:
            raise ValueError("No model has been built yet. Call build_mnist_model() first.")
        return self._model


# =============================================================================
# MNIST MODEL DEFINITION
# =============================================================================

class MNISTModel(nn.Module):
    """
    MNIST classification model.
    Implements the CNN architecture aligned with the reference Net class structure.
    """
    
    def __init__(self, config: ModelConfig):
        super(MNISTModel, self).__init__()
        self.config = config
        
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(14),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(14),
            nn.ReLU()
        ) # output_size = 28
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(14),
            nn.ReLU()
        ) # output_size = 28

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14
        #self.dropout1 = nn.Dropout(p=config.dropout_rate)

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(28),
            nn.ReLU()
        ) # output_size = 14
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(28),
            nn.ReLU()
        ) # output_size = 14

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7
        #self.dropout2 = nn.Dropout(p=config.dropout_rate)

        # CONVOLUTION BLOCK 3
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 5
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 3

       
        
        # Fully connected layers
        self.fc0 = nn.Linear(3*3*8, 20)
        self.fc1 = nn.Linear(20, 10)

    def forward(self, x):
        # Input Block
        x = self.convblock1(x)
        
        # Convolution Block 1
        x = self.convblock2(x)
        x = self.convblock3(x)
        
        # Transition Block 1
        x = self.pool1(x)
        #x = self.dropout1(x)
        
        # Convolution Block 2
        x = self.convblock4(x)
        x = self.convblock5(x)
        
        # Transition Block 2
        x = self.pool2(x)

        
        # Convolution Block 3
        x = self.convblock6(x)
        x = self.convblock7(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = torch.relu(self.fc0(x))
        x = self.fc1(x)
        
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
