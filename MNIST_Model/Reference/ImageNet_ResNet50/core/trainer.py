"""
Training System for ImageNet ResNet-50
Supports gradient accumulation, mixed precision, and checkpointing.

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ExponentialLR, 
    ReduceLROnPlateau, OneCycleLR
)
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, List
import os
from datetime import datetime
import glob

from .config import TrainingConfig
from .logger import Logger


class TrainingMetrics:
    """Tracks training metrics across epochs."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.train_top5_accuracies: List[float] = []
        self.test_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.test_top5_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.best_test_accuracy: float = 0.0
        self.best_test_top5_accuracy: float = 0.0
        self.best_epoch: int = 0
    
    def add_epoch_metrics(self, train_loss: float, train_acc: float, train_acc_top5: float,
                         test_loss: float, test_acc: float, test_acc_top5: float, lr: float):
        """Add metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.train_top5_accuracies.append(train_acc_top5)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        self.test_top5_accuracies.append(test_acc_top5)
        self.learning_rates.append(lr)
        
        # Update best metrics
        if test_acc > self.best_test_accuracy:
            self.best_test_accuracy = test_acc
            self.best_test_top5_accuracy = test_acc_top5
            self.best_epoch = len(self.test_accuracies)
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        if not self.test_accuracies:
            return {}
        
        return {
            'final_train_loss': self.train_losses[-1],
            'final_test_loss': self.test_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_test_accuracy': self.test_accuracies[-1],
            'final_train_top5_accuracy': self.train_top5_accuracies[-1],
            'final_test_top5_accuracy': self.test_top5_accuracies[-1],
            'best_test_accuracy': self.best_test_accuracy,
            'best_test_top5_accuracy': self.best_test_top5_accuracy,
            'best_epoch': self.best_epoch,
            'final_lr': self.learning_rates[-1]
        }
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



class ImageNetTrainer:
    """
    Trainer for ImageNet ResNet-50 with gradient accumulation and mixed precision.
    """
    
    def __init__(self, config: TrainingConfig, logger: Logger, model_save_dir: str):
        self.config = config
        self.logger = logger
        self.model_save_dir = model_save_dir
        self.metrics = TrainingMetrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if config.use_amp else None
        self.start_epoch = 0

         # Create loss function
        self.criterion = self.create_loss_function()
        
        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            if config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Loss function: {config.loss_function}")
        self.logger.info(f"Mixed Precision (AMP): {config.use_amp}")
        self.logger.info(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    
    def create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_fn = self.config.loss_function
        
        if loss_fn == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        
        elif loss_fn == 'NLLLoss':
            # Requires log_softmax in model output
            return nn.NLLLoss(
                reduction='mean'
            )
        
        elif loss_fn == 'BCEWithLogitsLoss':
            # For multi-label classification
            pos_weight = None
            if self.config.bce_pos_weight is not None:
                pos_weight = torch.tensor([self.config.bce_pos_weight]).to(self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        elif loss_fn == 'FocalLoss':
            alpha = self.config.focal_alpha
            if alpha is not None:
                alpha = torch.tensor([alpha]).to(self.device)
            return FocalLoss(
                alpha=alpha,
                gamma=self.config.focal_gamma,
                reduction='mean'
            )
        
        elif loss_fn == 'MSELoss':
            # For regression tasks or distillation
            return nn.MSELoss()
        
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type == 'SGD':
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        elif self.config.optimizer_type == 'Adam':
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'AdamW':
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        steps_per_epoch: int) -> Any:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'OneCycleLR':
            # Adjust for gradient accumulation
            effective_steps = steps_per_epoch // self.config.gradient_accumulation_steps
            
            return OneCycleLR(
                optimizer,
                max_lr=self.config.onecycle_max_lr,
                steps_per_epoch=effective_steps,
                epochs=self.config.epochs,
                pct_start=self.config.onecycle_pct_start,
                div_factor=self.config.onecycle_div_factor,
                final_div_factor=self.config.onecycle_final_div_factor,
                anneal_strategy=self.config.onecycle_anneal_strategy
            )
        elif self.config.scheduler_type == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.cosine_t_max,
                eta_min=self.config.cosine_eta_min
            )
        elif self.config.scheduler_type == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=self.config.plateau_mode,
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                threshold=self.config.plateau_threshold
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, scheduler: Any, epoch: int) -> Tuple[float, float, float]:
        """
        Train for one epoch with gradient accumulation and mixed precision.
        
        Returns:
            (avg_loss, top1_accuracy, top5_accuracy)
        """
        model.train()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        # Zero gradients at start
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    outputs = model(images)
                    
                    # Handle different loss functions
                    if self.config.loss_function == 'BCEWithLogitsLoss':
                        # Convert targets to one-hot for BCE
                        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
                        loss = self.criterion(outputs, targets_one_hot)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = model(images)
                
                # Handle different loss functions
                if self.config.loss_function == 'BCEWithLogitsLoss':
                    targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
                    loss = self.criterion(outputs, targets_one_hot)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or \
               (batch_idx + 1) == len(train_loader):
                
                # Gradient clipping
                if self.config.max_grad_norm is not None:
                    if self.config.use_amp:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update OneCycleLR scheduler per effective step
                if self.config.scheduler_type == 'OneCycleLR':
                    scheduler.step()
            
            # Calculate accuracies
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            
            correct_top1 += correct[:1].sum().item()
            correct_top5 += correct[:5].sum().item()
            total += targets.size(0)
            running_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch} - Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}, '
                f'Top-1: {100.0 * correct_top1 / total:.2f}%, '
                f'Top-5: {100.0 * correct_top5 / total:.2f}%'
            )
        
        avg_loss = running_loss / len(train_loader)
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total
        
        return avg_loss, top1_acc, top5_acc
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate for one epoch.
        
        Returns:
            (avg_loss, top1_accuracy, top5_accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        outputs = model(images)
                        
                        # Handle different loss functions
                        if self.config.loss_function == 'BCEWithLogitsLoss':
                            targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
                            loss = self.criterion(outputs, targets_one_hot)
                        else:
                            loss = self.criterion(outputs, targets)
                        
                        # Sum for proper averaging
                        loss = loss * targets.size(0)
                else:
                    outputs = model(images)
                    
                    if self.config.loss_function == 'BCEWithLogitsLoss':
                        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
                        loss = self.criterion(outputs, targets_one_hot)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    loss = loss * targets.size(0)
                
                # Calculate accuracies
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                
                correct_top1 += correct[:1].sum().item()
                correct_top5 += correct[:5].sum().item()
                total += targets.size(0)
                running_loss += loss.item()
        
        avg_loss = running_loss / total
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total
        
        return avg_loss, top1_acc, top5_acc
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Any, epoch: int, is_best: bool = False) -> str:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': self.metrics.get_final_metrics(),
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Save checkpoint
        if is_best:
            filepath = os.path.join(self.model_save_dir, 'best_model.pth')
        else:
            filepath = os.path.join(self.model_save_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, filepath)
        self.logger.log_checkpoint_save(epoch, filepath, is_best)
        
        # Keep only best N checkpoints (not including best_model.pth)
        if not is_best:
            self._cleanup_old_checkpoints()
        
        return filepath
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoint_pattern = os.path.join(self.model_save_dir, 'checkpoint_epoch_*.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) > self.config.keep_best_n_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=os.path.getmtime)
            # Remove oldest
            for old_checkpoint in checkpoints[:-self.config.keep_best_n_checkpoints]:
                os.remove(old_checkpoint)
                self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Any, checkpoint_path: str) -> int:
        """
        Load checkpoint and resume training.
        
        Returns:
            epoch to resume from
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {})
        
        self.logger.log_checkpoint_load(checkpoint_path, epoch, metrics)
        
        return epoch
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
             val_loader: DataLoader) -> TrainingMetrics:
        """
        Complete training loop with gradient accumulation and checkpointing.
        """
        self.logger.info("Initializing training...")
        
        # Setup optimizer and scheduler
        optimizer = self.create_optimizer(model)
        steps_per_epoch = len(train_loader)
        scheduler = self.create_scheduler(optimizer, steps_per_epoch)
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.start_epoch = self.load_checkpoint(
                model, optimizer, scheduler, self.config.resume_from_checkpoint
            )
            self.logger.info(f"Resuming from epoch {self.start_epoch + 1}")
        
        # Training loop
        best_accuracy = 0.0
        epochs_no_improve = 0
        
        for epoch in range(self.start_epoch, self.config.epochs):
            current_epoch = epoch + 1
            self.logger.log_epoch_start(current_epoch, self.config.epochs)
            
            # Train
            train_loss, train_acc, train_acc_top5 = self.train_epoch(
                model, train_loader, optimizer, scheduler, current_epoch
            )
            
            # Validate
            val_loss, val_acc, val_acc_top5 = self.validate_epoch(model, val_loader)
            
            # Update non-OneCycleLR schedulers
            if self.config.scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(val_acc if self.config.plateau_mode == 'max' else val_loss)
            elif self.config.scheduler_type not in ['OneCycleLR']:
                scheduler.step()
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.metrics.add_epoch_metrics(
                train_loss, train_acc, train_acc_top5,
                val_loss, val_acc, val_acc_top5, current_lr
            )
            
            # Log epoch results
            acc_diff = train_acc - val_acc
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
            
            self.logger.log_epoch_results(
                current_epoch, train_loss, train_acc, train_acc_top5,
                val_loss, val_acc, val_acc_top5, current_lr, acc_diff, gpu_mem
            )
            
            # Save checkpoint
            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc
                epochs_no_improve = 0
                self.save_checkpoint(model, optimizer, scheduler, current_epoch, is_best=True)
            else:
                epochs_no_improve += 1
            
            # Save periodic checkpoint
            if current_epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(model, optimizer, scheduler, current_epoch, is_best=False)
            
            # Early stopping check
            if epochs_no_improve >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
                break
        
        # Log final results
        final_metrics = self.metrics.get_final_metrics()
        self.logger.log_training_complete(final_metrics)
        
        return self.metrics
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        return allocated, reserved

