"""
Logging System for ImageNet Training
Singleton logger with file and console handlers.

Author: Krishnakanth
Date: 2025-10-13
"""

import logging
import os
from datetime import datetime
from typing import Optional
from .config import Config, LoggingConfig


class Logger:
    """
    Singleton logger class for centralized logging.
    Ensures only one logger instance exists throughout the application.
    """
    _instance: Optional['Logger'] = None
    _initialized: bool = False
    
    def __new__(cls, config: Optional[LoggingConfig] = None) -> 'Logger':
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        if not self._initialized and config is not None:
            self.config = config
            self._setup_logger()
            self._initialized = True
    
    def _setup_logger(self) -> None:
        """Setup the logger with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger('ImageNet_Training')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure log directory exists
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
        except Exception as e:
            # If directory creation fails, continue and rely on console logging
            # We don't call logger.* yet because handlers aren't set up
            pass
        
        # File handler (optional - fall back to console only if file can't be created)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(
            self.config.log_dir, 
            f"{timestamp}_{self.config.log_file}"
        )
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            # If file handler cannot be created (e.g., permissions), continue with console handler only
            pass
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add console handler (always)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file_path}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def log_experiment_start(self, config: Config) -> None:
        """Log experiment start with configuration."""
        self.info("=" * 70)
        self.info("IMAGENET RESNET-50 TRAINING EXPERIMENT STARTED")
        self.info("=" * 70)
        self.info(f"Data Config: {config.data}")
        self.info(f"Model Config: {config.model}")
        self.info(f"Training Config: {config.training}")
        self.info("=" * 70)
    
    def log_config_summary(self, config: Config) -> None:
        """Log configuration summary."""
        self.info("Configuration Summary:")
        self.info(f"  Dataset: ImageNet-1k")
        self.info(f"  Model: {config.model.model_name}")
        self.info(f"  Input Size: {config.data.input_size}x{config.data.input_size}")
        self.info(f"  Num Classes: {config.model.num_classes}")
        
        # Data configuration
        self.info(f"  Batch Size: {config.data.batch_size}")
        self.info(f"  Gradient Accumulation: {config.training.gradient_accumulation_steps}")
        self.info(f"  Effective Batch Size: {config.get_effective_batch_size()}")
        self.info(f"  Num Workers: {config.data.num_workers}")
        
        # Subset training
        if config.data.use_subset:
            self.info(f"  Using Subset: {config.data.subset_percentage * 100}% (stratified)")
        
        # Training configuration
        self.info(f"  Epochs: {config.training.epochs}")
        self.info(f"  Learning Rate: {config.training.learning_rate}")
        self.info(f"  Optimizer: {config.training.optimizer_type}")
        self.info(f"  Scheduler: {config.training.scheduler_type}")
        self.info(f"  Weight Decay: {config.training.weight_decay}")
        self.info(f"  Mixed Precision: {config.training.use_amp}")
        self.info(f"  Gradient Clipping: {config.training.max_grad_norm}")
        self.info(f"  Label Smoothing: {config.training.label_smoothing}")
        
        # Model configuration
        self.info(f"  Dropout Rate: {config.model.dropout_rate}")
        self.info(f"  Use Pretrained: {config.model.use_pretrained}")
        
        self.info("=" * 70)
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.info(f"Starting Epoch {epoch}/{total_epochs}")
    
    def log_epoch_results(self, epoch: int, train_loss: float, train_acc: float,
                         train_acc_top5: float, test_loss: float, test_acc: float,
                         test_acc_top5: float, current_lr: float, 
                         acc_diff: float = None, gpu_memory_mb: float = None) -> None:
        """Log comprehensive epoch results."""
        msg = (f"Epoch {epoch:3d}: "
               f"Train Loss: {train_loss:.4f}, Train Top-1: {train_acc:.2f}%, Train Top-5: {train_acc_top5:.2f}% | "
               f"Test Loss: {test_loss:.4f}, Test Top-1: {test_acc:.2f}%, Test Top-5: {test_acc_top5:.2f}% | "
               f"LR: {current_lr:.6f}")
        
        if acc_diff is not None:
            msg += f" | Gap: {acc_diff:.2f}%"
        
        if gpu_memory_mb is not None:
            msg += f" | GPU: {gpu_memory_mb:.0f}MB"
        
        self.info(msg)
    
    def log_checkpoint_save(self, epoch: int, filepath: str, is_best: bool = False) -> None:
        """Log checkpoint save."""
        checkpoint_type = "BEST MODEL" if is_best else f"Checkpoint (Epoch {epoch})"
        self.info(f"Saved {checkpoint_type}: {filepath}")
    
    def log_checkpoint_load(self, filepath: str, epoch: int, metrics: dict) -> None:
        """Log checkpoint load."""
        self.info(f"Loaded checkpoint from: {filepath}")
        self.info(f"  Resume from epoch: {epoch}")
        self.info(f"  Previous best accuracy: {metrics.get('best_test_accuracy', 'N/A')}")
    
    def log_training_complete(self, final_metrics: dict) -> None:
        """Log training completion summary."""
        self.info("=" * 70)
        self.info("TRAINING COMPLETED SUCCESSFULLY")
        self.info("=" * 70)
        self.info(f"Final Train Loss: {final_metrics.get('final_train_loss', 0):.4f}")
        self.info(f"Final Test Loss: {final_metrics.get('final_test_loss', 0):.4f}")
        self.info(f"Final Train Top-1 Acc: {final_metrics.get('final_train_accuracy', 0):.2f}%")
        self.info(f"Final Test Top-1 Acc: {final_metrics.get('final_test_accuracy', 0):.2f}%")
        self.info(f"Best Test Top-1 Acc: {final_metrics.get('best_test_accuracy', 0):.2f}%")
        self.info(f"Final Test Top-5 Acc: {final_metrics.get('final_test_top5_accuracy', 0):.2f}%")
        self.info(f"Best Test Top-5 Acc: {final_metrics.get('best_test_top5_accuracy', 0):.2f}%")
        self.info("=" * 70)
    
    def log_gpu_memory(self, allocated_mb: float, reserved_mb: float) -> None:
        """Log GPU memory usage."""
        self.debug(f"GPU Memory - Allocated: {allocated_mb:.0f}MB, Reserved: {reserved_mb:.0f}MB")

