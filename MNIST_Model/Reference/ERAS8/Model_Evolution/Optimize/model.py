"""
CIFAR100 Model Definition
Contains the model architecture classes for CIFAR100 classification.

This module provides:
- ModelConfig: Configuration for model architecture
- ModelBuilder: Builder pattern for constructing models
- CIFAR100Model: The main CNN model class

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
    input_channels: int = 3
    input_size: Tuple[int, int] = (32, 32)
    num_classes: int = 100
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
    
    def build_cifar100_model(self, config: ModelConfig) -> 'CIFAR100Model':
        """Build CIFAR100 classification model."""
        self._model = CIFAR100Model(config)
        return self._model
    
    def get_model(self) -> 'CIFAR100Model':
        """Get the built model."""
        if self._model is None:
            raise ValueError("No model has been built yet. Call build_cifar100_model() first.")
        return self._model


# =============================================================================
# CIFAR100 MODEL DEFINITION
# =============================================================================
class CIFAR100Model(nn.Module):
    """
    CIFAR100 classification model.
    Improved architecture with proper Batch Normalization and regularization.
    """
    
    def __init__(self, config: ModelConfig):
        super(CIFAR100Model, self).__init__()
        self.config = config
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.config.input_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),nn.ReLU(inplace=True))
           
        self.res1 = nn.Sequential(self.conv_block(64, 64,activation=True), self.conv_block(64, 64))
        self.res2 = nn.Sequential(self.conv_block(64, 64,activation=True), self.conv_block(64, 64))
        self.res3 = nn.Sequential(self.conv_block(64, 64,activation=True), self.conv_block(64, 64))
        self.downsample1=nn.Sequential(self.conv_block(64, 128,pool=True)) 
        self.res4 = nn.Sequential(self.conv_block(64, 128,activation=True, pool=True),
                                  self.conv_block(128,128))
        self.res5 = nn.Sequential(self.conv_block(128, 128,activation=True), self.conv_block(128, 128))
        self.res6 = nn.Sequential(self.conv_block(128, 128,activation=True), self.conv_block(128, 128))
        self.res7 = nn.Sequential(self.conv_block(128, 128,activation=True), self.conv_block(128, 128))
        self.res8 = nn.Sequential(self.conv_block(128, 256,activation=True, pool=True),
                                  self.conv_block(256,256))
        self.downsample2 = nn.Sequential(self.conv_block(128, 256,pool=True))
        self.res9 = nn.Sequential(self.conv_block(256, 256,activation=True), self.conv_block(256, 256))
        self.res10 = nn.Sequential(self.conv_block(256, 256,activation=True), self.conv_block(256, 256))
        self.res11 = nn.Sequential(self.conv_block(256, 256,activation=True), self.conv_block(256, 256))
        self.res12 = nn.Sequential(self.conv_block(256, 256,activation=True), self.conv_block(256, 256))
        self.res13 = nn.Sequential(self.conv_block(256, 256,activation=True), self.conv_block(256, 256))
        self.res14 = nn.Sequential(self.conv_block(256, 512,activation=True, pool=True),
                                   self.conv_block(512,512))
        
        self.downsample3 = nn.Sequential(self.conv_block(256, 512,pool=True))
        self.res15 = nn.Sequential(self.conv_block(512, 512,activation=True), self.conv_block(512, 512))
        self.res16 = nn.Sequential(self.conv_block(512, 512,activation=True), self.conv_block(512, 512,activation=True))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        #nn.Dropout(0.2),
                                        nn.Linear(512,1000))
        self.apply(self.init_weights)
    
    def conv_block(self, in_channels, out_channels, activation=True, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + out
        out = self.downsample1(out) +self.res4(out)
        out = self.res5(out) + out
        out = self.res6(out) + out
        out = self.res7(out) + out
        out = self.downsample2(out) +self.res8(out)
        out = self.res9(out) + out
        out = self.res10(out) + out
        out = self.res11(out) + out
        out = self.res12(out) + out
        out = self.res13(out) + out
        out = self.downsample3(out) + self.res14(out) 
        out = self.res15(out) + out
        out = self.res16(out) + out
        out = self.classifier(out)
        return F.log_softmax(out, dim=-1)
    
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
