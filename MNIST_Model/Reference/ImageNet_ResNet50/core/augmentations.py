"""
Advanced Augmentation Utilities for ImageNet Training
Implements Mixup, Cutmix, and other advanced augmentation techniques.

Author: Krishnakanth
Date: 2025-10-16
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class MixupCutmix:
    """
    Implements Mixup and Cutmix augmentations.
    Can be applied dynamically during training.
    """
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                 mixup_prob: float = 0.5, cutmix_prob: float = 0.5):
        """
        Initialize Mixup/Cutmix augmentation.
        
        Args:
            mixup_alpha: Mixup alpha parameter for beta distribution
            cutmix_alpha: Cutmix alpha parameter for beta distribution
            mixup_prob: Probability of applying mixup
            cutmix_prob: Probability of applying cutmix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
    
    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply Mixup or Cutmix augmentation.
        
        Args:
            images: Batch of images [B, C, H, W]
            targets: Batch of targets [B]
        
        Returns:
            augmented_images, targets_a, targets_b, lambda_value
            where final_target = lambda * targets_a + (1 - lambda) * targets_b
        """
        batch_size = images.size(0)
        
        # Decide which augmentation to apply
        use_mixup = np.random.rand() < self.mixup_prob
        use_cutmix = np.random.rand() < self.cutmix_prob
        
        # If both are selected, randomly choose one
        if use_mixup and use_cutmix:
            use_mixup = np.random.rand() < 0.5
            use_cutmix = not use_mixup
        
        # Generate random indices for mixing
        indices = torch.randperm(batch_size).to(images.device)
        shuffled_images = images[indices]
        shuffled_targets = targets[indices]
        
        if use_mixup:
            # Mixup: Blend images and labels
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * shuffled_images
            return mixed_images, targets, shuffled_targets, lam
        
        elif use_cutmix:
            # Cutmix: Cut and paste regions
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            
            _, _, H, W = images.size()
            cut_ratio = np.sqrt(1.0 - lam)
            cut_h = int(H * cut_ratio)
            cut_w = int(W * cut_ratio)
            
            # Random center for cut region
            cx = np.random.randint(H)
            cy = np.random.randint(W)
            
            # Calculate bounding box
            bbx1 = np.clip(cx - cut_h // 2, 0, H)
            bby1 = np.clip(cy - cut_w // 2, 0, W)
            bbx2 = np.clip(cx + cut_h // 2, 0, H)
            bby2 = np.clip(cy + cut_w // 2, 0, W)
            
            # Apply cutmix
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual cut area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
            
            return mixed_images, targets, shuffled_targets, lam
        
        else:
            # No augmentation
            return images, targets, targets, 1.0


def mixup_criterion(criterion, pred: torch.Tensor, targets_a: torch.Tensor,
                   targets_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Calculate loss for mixed samples.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        targets_a: First set of targets
        targets_b: Second set of targets
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """
    Label smoothing cross entropy loss.
    Useful when combined with Mixup/Cutmix.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, num_classes] (logits)
            target: Ground truth labels [B]
        """
        num_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(log_pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / num_classes
        
        # Calculate loss
        loss = (-targets_smooth * log_pred).sum(dim=-1).mean()
        
        return loss

