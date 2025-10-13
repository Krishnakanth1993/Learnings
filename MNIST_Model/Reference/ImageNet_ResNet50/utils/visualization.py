"""
Visualization Utilities for Training
Training curves, confusion matrices, and sample predictions.

Author: Krishnakanth
Date: 2025-10-13
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import seaborn as sns
from pathlib import Path


class TrainingVisualizer:
    """Visualize training progress and results."""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float],
                            train_accs: List[float], val_accs: List[float],
                            train_top5_accs: List[float], val_top5_accs: List[float],
                            learning_rates: List[float],
                            save_path: Optional[str] = None):
        """
        Plot comprehensive training curves.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            train_accs: Training top-1 accuracies
            val_accs: Validation top-1 accuracies
            train_top5_accs: Training top-5 accuracies
            val_top5_accs: Validation top-5 accuracies
            learning_rates: Learning rates per epoch
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle('ImageNet ResNet-50 Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top-1 Accuracy
        axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Top-1', linewidth=2)
        axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Top-1', linewidth=2)
        axes[0, 1].set_title('Top-1 Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[0, 2].plot(epochs, train_top5_accs, 'b-', label='Train Top-5', linewidth=2)
        axes[0, 2].plot(epochs, val_top5_accs, 'r-', label='Val Top-5', linewidth=2)
        axes[0, 2].set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy Gap (Train - Val)
        acc_gap = [t - v for t, v in zip(train_accs, val_accs)]
        axes[1, 1].plot(epochs, acc_gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Overfitting Threshold')
        axes[1, 1].set_title('Generalization Gap (Train - Val)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Training Summary Stats
        axes[1, 2].axis('off')
        summary_text = (
            f"Training Summary\n"
            f"{'='*40}\n"
            f"Total Epochs: {len(epochs)}\n"
            f"\n"
            f"Final Train Top-1: {train_accs[-1]:.2f}%\n"
            f"Final Val Top-1: {val_accs[-1]:.2f}%\n"
            f"Best Val Top-1: {max(val_accs):.2f}% (Epoch {val_accs.index(max(val_accs))+1})\n"
            f"\n"
            f"Final Train Top-5: {train_top5_accs[-1]:.2f}%\n"
            f"Final Val Top-5: {val_top5_accs[-1]:.2f}%\n"
            f"Best Val Top-5: {max(val_top5_accs):.2f}%\n"
            f"\n"
            f"Final LR: {learning_rates[-1]:.6f}\n"
            f"Final Gap: {acc_gap[-1]:.2f}%\n"
        )
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None,
                             normalize: bool = False, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (20, 20), top_n_classes: int = 50):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Whether to normalize by row
            save_path: Path to save figure
            figsize: Figure size
            top_n_classes: Show only top N classes (for readability with 1000 classes)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # For ImageNet with 1000 classes, show subset
        if cm.shape[0] > top_n_classes:
            # Show top N most confused classes
            confusion_scores = cm.sum(axis=1) - np.diag(cm)  # Total errors per class
            top_indices = np.argsort(confusion_scores)[-top_n_classes:]
            cm = cm[np.ix_(top_indices, top_indices)]
            
            if class_names:
                class_names = [class_names[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd',
                   cmap='Blues', square=True, cbar_kws={'label': 'Count'},
                   xticklabels=class_names if class_names and len(class_names) <= 50 else False,
                   yticklabels=class_names if class_names and len(class_names) <= 50 else False)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        if cm.shape[0] < 1000:
            title += f' (Top {cm.shape[0]} Confused Classes)'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_per_class_accuracy(per_class_acc: np.ndarray, class_names: List[str] = None,
                               save_path: Optional[str] = None, top_n: int = 50):
        """
        Plot per-class accuracy bar chart.
        
        Args:
            per_class_acc: Per-class accuracy array
            class_names: List of class names
            save_path: Path to save figure
            top_n: Show top and bottom N classes
        """
        # Sort by accuracy
        sorted_indices = np.argsort(per_class_acc)
        
        # Show worst N and best N classes
        indices_to_show = np.concatenate([sorted_indices[:top_n], sorted_indices[-top_n:]])
        accs_to_show = per_class_acc[indices_to_show]
        
        if class_names:
            names_to_show = [class_names[i] for i in indices_to_show]
        else:
            names_to_show = [f"Class {i}" for i in indices_to_show]
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        colors = ['red' if acc < 50 else 'orange' if acc < 70 else 'green' 
                 for acc in accs_to_show]
        
        bars = ax.barh(range(len(accs_to_show)), accs_to_show, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names_to_show)))
        ax.set_yticklabels(names_to_show, fontsize=8)
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Per-Class Accuracy (Bottom {top_n} and Top {top_n} Classes)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, accs_to_show)):
            ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-class accuracy plot saved to: {save_path}")
        
        return fig

