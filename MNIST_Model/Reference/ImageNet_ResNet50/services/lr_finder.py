"""
Learning Rate Finder Service
Helps find optimal learning rate range before training.

Based on the LR range test method from Leslie Smith's paper:
"Cyclical Learning Rates for Training Neural Networks"

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import copy


class LRFinder:
    """
    Learning Rate Finder using exponential LR increase.
    Helps identify optimal learning rate range before full training.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 criterion: nn.Module, device: torch.device):
        """
        Initialize LR Finder.
        
        Args:
            model: Neural network model
            optimizer: Optimizer (will be reset during search)
            criterion: Loss criterion
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save initial model and optimizer state
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    def range_test(self, train_loader: DataLoader, start_lr: float = 1e-7,
                   end_lr: float = 10, num_iter: int = 100,
                   smooth_f: float = 0.05) -> Tuple[list, list]:
        """
        Perform LR range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            smooth_f: Smoothing factor for loss (0 = no smoothing)
        
        Returns:
            (learning_rates, losses) lists
        """
        # Reset to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        # Calculate LR multiplier per iteration
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # Set starting LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        self.model.train()
        
        learning_rates = []
        losses = []
        best_loss = float('inf')
        avg_loss = 0.0
        batch_num = 0
        
        print(f"Running LR range test from {start_lr:.2e} to {end_lr:.2e}")
        
        iterator = iter(train_loader)
        pbar = tqdm(range(num_iter), desc='LR Finder')
        
        for iteration in pbar:
            # Get batch
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update LR for next iteration
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Smooth loss
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            
            losses.append(avg_loss)
            
            # Check for divergence
            if avg_loss > 4 * best_loss or torch.isnan(loss):
                print(f"\nStopping early at iteration {iteration}: Loss diverged")
                break
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Increase LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_mult
            
            pbar.set_description(f'LR: {current_lr:.2e}, Loss: {avg_loss:.4f}')
            
            batch_num += 1
        
        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        print(f"\nLR Finder completed. Tested {len(learning_rates)} iterations.")
        
        return learning_rates, losses
    
    def plot(self, learning_rates: list, losses: list,
             save_path: Optional[str] = None, suggest_lr: bool = True) -> Tuple[Optional[float], Optional[float]]:
        """
        Plot LR finder results and suggest optimal LR.
        
        Args:
            learning_rates: List of learning rates tested
            losses: Corresponding losses
            save_path: Path to save plot
            suggest_lr: Whether to suggest optimal LR
        
        Returns:
            (suggested_lr, suggested_max_lr) if suggest_lr else (None, None)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(learning_rates, losses, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        suggested_lr = None
        suggested_max_lr = None
        
        if suggest_lr and len(losses) > 10:
            # Find steepest gradient (fastest learning)
            gradients = np.gradient(losses)
            min_grad_idx = np.argmin(gradients)
            suggested_lr = learning_rates[min_grad_idx]
            
            # Find LR at minimum loss
            min_loss_idx = np.argmin(losses)
            suggested_max_lr = learning_rates[min_loss_idx]
            
            # Plot suggestions
            ax.axvline(x=suggested_lr, color='r', linestyle='--', 
                      label=f'Suggested LR: {suggested_lr:.2e}', linewidth=2)
            ax.axvline(x=suggested_max_lr, color='g', linestyle='--',
                      label=f'Max LR: {suggested_max_lr:.2e}', linewidth=2)
            ax.legend(fontsize=10)
            
            print(f"\nSuggested Learning Rate: {suggested_lr:.2e}")
            print(f"Suggested Max LR (for OneCycleLR): {suggested_max_lr:.2e}")
            print(f"\nRecommendation:")
            print(f"  - Use LR: {suggested_lr:.2e} for Adam/AdamW")
            print(f"  - Use max_lr: {suggested_max_lr / 10:.2e} for OneCycleLR")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"LR finder plot saved to: {save_path}")
        
        plt.show()
        
        return suggested_lr, suggested_max_lr
    
    def find_lr(self, train_loader: DataLoader, start_lr: float = 1e-7,
                end_lr: float = 10, num_iter: int = 100,
                save_path: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Complete LR finding process with plotting.
        
        Returns:
            (suggested_lr, suggested_max_lr)
        """
        learning_rates, losses = self.range_test(
            train_loader, start_lr, end_lr, num_iter
        )
        
        suggested_lr, suggested_max_lr = self.plot(
            learning_rates, losses, save_path, suggest_lr=True
        )
        
        return suggested_lr, suggested_max_lr

