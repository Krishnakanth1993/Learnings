"""
CIFAR100 Custom ResNet Model Definition
Matches the architecture used for training cifar100_model.pth

Author: Krishnakanth
Date: 2025-10-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_channels: int = 3
    input_size: Tuple[int, int] = (32, 32)
    num_classes: int = 100
    dropout_rate: float = 0.05


class CIFAR100Model(nn.Module):
    """
    CIFAR100 classification model - Custom ResNet architecture.
    This matches the architecture of your trained model (23M parameters).
    """
    
    def __init__(self, config: ModelConfig):
        super(CIFAR100Model, self).__init__()
        self.config = config
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res1 = nn.Sequential(self.conv_block(64, 64, activation=True), self.conv_block(64, 64))
        self.res2 = nn.Sequential(self.conv_block(64, 64, activation=True), self.conv_block(64, 64))
        self.res3 = nn.Sequential(self.conv_block(64, 64, activation=True), self.conv_block(64, 64))
        
        self.downsample1 = nn.Sequential(self.conv_block(64, 128, pool=True))
        self.res4 = nn.Sequential(
            self.conv_block(64, 128, activation=True, pool=True),
            self.conv_block(128, 128)
        )
        self.res5 = nn.Sequential(self.conv_block(128, 128, activation=True), self.conv_block(128, 128))
        self.res6 = nn.Sequential(self.conv_block(128, 128, activation=True), self.conv_block(128, 128))
        self.res7 = nn.Sequential(self.conv_block(128, 128, activation=True), self.conv_block(128, 128))
        
        self.res8 = nn.Sequential(
            self.conv_block(128, 256, activation=True, pool=True),
            self.conv_block(256, 256)
        )
        self.downsample2 = nn.Sequential(self.conv_block(128, 256, pool=True))
        self.res9 = nn.Sequential(self.conv_block(256, 256, activation=True), self.conv_block(256, 256))
        self.res10 = nn.Sequential(self.conv_block(256, 256, activation=True), self.conv_block(256, 256))
        self.res11 = nn.Sequential(self.conv_block(256, 256, activation=True), self.conv_block(256, 256))
        self.res12 = nn.Sequential(self.conv_block(256, 256, activation=True), self.conv_block(256, 256))
        self.res13 = nn.Sequential(self.conv_block(256, 256, activation=True), self.conv_block(256, 256))
        
        self.res14 = nn.Sequential(
            self.conv_block(256, 512, activation=True, pool=True),
            self.conv_block(512, 512)
        )
        self.downsample3 = nn.Sequential(self.conv_block(256, 512, pool=True))
        self.res15 = nn.Sequential(self.conv_block(512, 512, activation=True), self.conv_block(512, 512))
        self.res16 = nn.Sequential(
            self.conv_block(512, 512, activation=True),
            self.conv_block(512, 512, activation=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1000)  # Original has 1000, we'll fix this
        )
        
        self.apply(self.init_weights)
    
    def conv_block(self, in_channels, out_channels, activation=True, pool=False):
        """Create a convolutional block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + out
        out = self.downsample1(out) + self.res4(out)
        out = self.res5(out) + out
        out = self.res6(out) + out
        out = self.res7(out) + out
        out = self.downsample2(out) + self.res8(out)
        out = self.res9(out) + out
        out = self.res10(out) + out
        out = self.res11(out) + out
        out = self.res12(out) + out
        out = self.res13(out) + out
        out = self.downsample3(out) + self.res14(out)
        out = self.res15(out) + out
        out = self.res16(out) + out
        out = self.classifier(out)
        
        # Get first 100 outputs (since classifier outputs 1000)
        return F.log_softmax(out[:, :100], dim=-1)


# Aliases for compatibility
CIFAR100ResNet34 = CIFAR100Model
CIFAR100ResNet18 = CIFAR100Model