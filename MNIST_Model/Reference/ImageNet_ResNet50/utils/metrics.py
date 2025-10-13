"""
Metrics Utilities for ImageNet Training
Top-K accuracy, confusion matrix, and performance tracking.

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
import numpy as np
from typing import Tuple, List
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class TopKAccuracy:
    """Calculate Top-K accuracy metrics."""
    
    @staticmethod
    def calculate(outputs: torch.Tensor, targets: torch.Tensor, 
                 topk: Tuple[int, ...] = (1, 5)) -> List[float]:
        """
        Calculate top-k accuracy.
        
        Args:
            outputs: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            topk: Tuple of k values to calculate
        
        Returns:
            List of top-k accuracies (percentages)
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)
            
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            
            return res
    
    @staticmethod
    def calculate_per_class(outputs: torch.Tensor, targets: torch.Tensor,
                           num_classes: int, k: int = 1) -> np.ndarray:
        """
        Calculate per-class top-k accuracy.
        
        Returns:
            Array of per-class accuracies
        """
        with torch.no_grad():
            _, pred = outputs.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            
            per_class_correct = np.zeros(num_classes)
            per_class_total = np.zeros(num_classes)
            
            for i in range(targets.size(0)):
                target_class = targets[i].item()
                per_class_total[target_class] += 1
                if correct[:k, i].any():
                    per_class_correct[target_class] += 1
            
            # Avoid division by zero
            per_class_acc = np.zeros(num_classes)
            mask = per_class_total > 0
            per_class_acc[mask] = 100.0 * per_class_correct[mask] / per_class_total[mask]
            
            return per_class_acc


class ConfusionMatrixCalculator:
    """Calculate and analyze confusion matrix."""
    
    @staticmethod
    def compute(predictions: np.ndarray, targets: np.ndarray, 
                num_classes: int) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predicted class labels
            targets: True class labels
            num_classes: Number of classes
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        return confusion_matrix(targets, predictions, labels=range(num_classes))
    
    @staticmethod
    def get_most_confused_pairs(cm: np.ndarray, top_n: int = 10) -> List[Tuple[int, int, int]]:
        """
        Find most confused class pairs.
        
        Args:
            cm: Confusion matrix
            top_n: Number of top confused pairs to return
        
        Returns:
            List of (true_class, predicted_class, count) tuples
        """
        # Zero out diagonal (correct predictions)
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # Find top confused pairs
        flat_indices = np.argsort(cm_copy.flatten())[::-1][:top_n]
        pairs = []
        
        for idx in flat_indices:
            true_idx = idx // cm.shape[1]
            pred_idx = idx % cm.shape[1]
            count = cm[true_idx, pred_idx]
            if count > 0:
                pairs.append((true_idx, pred_idx, int(count)))
        
        return pairs
    
    @staticmethod
    def get_classification_report(predictions: np.ndarray, targets: np.ndarray,
                                 class_names: List[str] = None) -> str:
        """
        Generate classification report.
        
        Args:
            predictions: Predicted class labels
            targets: True class labels
            class_names: List of class names
        
        Returns:
            Classification report string
        """
        return classification_report(
            targets, predictions, 
            target_names=class_names if class_names else None,
            zero_division=0
        )


class PerformanceTracker:
    """Track performance metrics over time."""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_top1': [],
            'train_top5': [],
            'val_loss': [],
            'val_top1': [],
            'val_top5': [],
            'learning_rate': []
        }
    
    def add_epoch(self, epoch: int, train_loss: float, train_top1: float, train_top5: float,
                 val_loss: float, val_top1: float, val_top5: float, lr: float):
        """Add epoch metrics to history."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_top1'].append(train_top1)
        self.history['train_top5'].append(train_top5)
        self.history['val_loss'].append(val_loss)
        self.history['val_top1'].append(val_top1)
        self.history['val_top5'].append(val_top5)
        self.history['learning_rate'].append(lr)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to pandas DataFrame."""
        return pd.DataFrame(self.history)
    
    def save_csv(self, filepath: str):
        """Save history to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

