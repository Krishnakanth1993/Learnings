"""
Data Manager for CIFAR-10 dataset
Author: Krishnakanth
Date: 29-09-2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
from io import StringIO
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = './data'
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    # CIFAR-100 specific normalization values
    cifar100_mean: Tuple[float, ...] = (0.507076, 0.486550, 0.440919)
    cifar100_std: Tuple[float, ...] = (0.267334, 0.256438, 0.276150)
    # Data augmentation
    rotation_range: Tuple[float, float] = (-7.0, 7.0)
    fill_value: int = 1


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 20
    learning_rate: float = 0.01
    momentum: float = 0.8
    weight_decay: float = 0.0
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    seed: int = 1
    
    # Optimizer configuration
    optimizer_type: str = 'SGD'  # Options: 'SGD', 'Adam', 'AdamW', 'RMSprop'
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    rmsprop_alpha: float = 0.99
    
    # Scheduler configuration
    scheduler_type: str = 'StepLR'  # Options: 'StepLR', 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau'
    cosine_t_max: int = 20
    exponential_gamma: float = 0.95
    plateau_mode: str = 'min'
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = None  # Will be set to script directory
    log_file: str = 'cifar100_training.log'
    log_level: str = 'INFO'
    save_model: bool = True
    model_save_dir: str = None  # Will be set to script directory


@dataclass
class Config:
    """Main configuration class that combines all configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    #model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set log and model directories relative to script location
        if self.logging.log_dir is None:
            self.logging.log_dir = os.path.join(script_dir, 'logs')
        if self.logging.model_save_dir is None:
            self.logging.model_save_dir = os.path.join(script_dir, 'models')
        
        # Create necessary directories
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.model_save_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)


# =============================================================================
# LOGGING SYSTEM (Singleton Pattern)
# =============================================================================

class Logger:
    """
    Singleton logger class for centralized logging.
    Implements the Singleton pattern to ensure only one logger instance.
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
        self.logger = logging.getLogger('CIFAR-100_Training')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file_path = os.path.join(
            self.config.log_dir, 
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.config.log_file}"
        )
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
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
        self.info("=" * 50)
        self.info("CIFAR-100 TRAINING EXPERIMENT STARTED")
        self.info("=" * 50)
        self.info(f"Data Config: {config.data}")
        self.info(f"Model Config: {config.model}")
        self.info(f"Training Config: {config.training}")
        self.info("=" * 50)
    
    def log_updated_config(self, config: Config) -> None:
        """Log the updated configuration with actual values from main()."""
        self.info("Updated Configuration (from main()):")
        self.info(f"  - Epochs: {config.training.epochs}")
        self.info(f"  - Learning Rate: {config.training.learning_rate}")
        self.info(f"  - Optimizer: {config.training.optimizer_type}")
        self.info(f"  - Weight Decay: {config.training.weight_decay}")
        
        # Optimizer-specific parameters
        if config.training.optimizer_type == 'SGD':
            self.info(f"  - Momentum: {config.training.momentum}")
        elif config.training.optimizer_type in ['Adam', 'AdamW']:
            self.info(f"  - Adam Betas: {config.training.adam_betas}")
            self.info(f"  - Adam Eps: {config.training.adam_eps}")
        elif config.training.optimizer_type == 'RMSprop':
            self.info(f"  - RMSprop Alpha: {config.training.rmsprop_alpha}")
        
        # Scheduler information
        self.info(f"  - Scheduler: {config.training.scheduler_type}")
        
        # Scheduler-specific parameters
        if config.training.scheduler_type == 'StepLR':
            self.info(f"  - Step Size: {config.training.scheduler_step_size}")
            self.info(f"  - Gamma: {config.training.scheduler_gamma}")
        elif config.training.scheduler_type == 'CosineAnnealingLR':
            self.info(f"  - T Max: {config.training.cosine_t_max}")
        elif config.training.scheduler_type == 'ExponentialLR':
            self.info(f"  - Gamma: {config.training.exponential_gamma}")
        elif config.training.scheduler_type == 'ReduceLROnPlateau':
            self.info(f"  - Mode: {config.training.plateau_mode}")
            self.info(f"  - Factor: {config.training.plateau_factor}")
            self.info(f"  - Patience: {config.training.plateau_patience}")
            self.info(f"  - Threshold: {config.training.plateau_threshold}")
        self.info(f"  - Batch Size: {config.data.batch_size}")
        self.info(f"  - Num Workers: {config.data.num_workers}")
        self.info(f"  - Pin Memory: {config.data.pin_memory}")
        self.info(f"  - Shuffle: {config.data.shuffle}")
        self.info(f"  - Dropout Rate: {config.model.dropout_rate}")
        self.info(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.info(f"  - Log Directory: {config.logging.log_dir}")
        self.info(f"  - Model Save Directory: {config.logging.model_save_dir}")
        self.info(f"  - Save Model: {config.logging.save_model}")
        self.info(f"  - Log Level: {config.logging.log_level}")
        self.info("=" * 50)
    
    def log_epoch_results(self, epoch: int, train_loss: float, train_acc: float, 
                         test_loss: float, test_acc: float, current_lr: float = None) -> None:
        """Log epoch results."""
        if current_lr is not None:
            self.info(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}")
        else:
            self.info(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                     f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    def log_model_summary(self, total_params: int, has_batchnorm: bool, 
                         has_dropout: bool, has_fc_or_gap: bool) -> None:
        """Log model summary."""
        self.info("Model Summary:")
        self.info(f"  - Total Parameters: {total_params:,}")
        self.info(f"  - Batch Normalization: {'Yes' if has_batchnorm else 'No'}")
        self.info(f"  - Dropout: {'Yes' if has_dropout else 'No'}")
        self.info(f"  - FC/GAP Layers: {'Yes' if has_fc_or_gap else 'No'}")
    
    def log_detailed_model_summary(self, summary_text: str) -> None:
        """Log the detailed model summary from torchsummary."""
        self.info("=" * 50)
        self.info("DETAILED MODEL ARCHITECTURE SUMMARY")
        self.info("=" * 50)
        for line in summary_text.splitlines():
            self.info(line)
        self.info("=" * 50)
    
    def log_epoch_results_with_difference(self, epoch: int, train_loss: float, train_acc: float, 
                                        test_loss: float, test_acc: float, current_lr: float, 
                                        acc_diff: float, overfitting_epochs: int) -> None:
        """Log epoch results with accuracy difference."""
        overfitting_warning = f" (OVERFITTING: {overfitting_epochs} epochs)" if overfitting_epochs > 0 else ""
        
        self.info(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                 f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                 f"Acc Diff: {acc_diff:.2f}%, LR: {current_lr:.6f}{overfitting_warning}")



# =============================================================================
# DATA LOADING SYSTEM (Factory & Strategy Patterns)
# =============================================================================

class TransformStrategy(ABC):
    """Abstract base class for transformation strategies."""
    
    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Get the composed transforms."""
        pass


class CIFAR100TransformStrategy(TransformStrategy):
    """Strategy for CIFAR-100 data transformations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_transforms(self) -> transforms.Compose:
        """Get CIFAR-100 specific transforms."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.cifar100_mean, self.config.cifar100_std)
        ])


class CIFAR100TrainTransformStrategy(TransformStrategy):
    """Strategy for CIFAR-100 training data transformations with augmentation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_transforms(self) -> transforms.Compose:
        """Get CIFAR-100 training transforms with augmentation."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.cifar100_mean, self.config.cifar100_std)
        ])


class CIFAR100TestTransformStrategy(TransformStrategy):
    """Strategy for CIFAR-100 test data transformations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_transforms(self) -> transforms.Compose:
        """Get CIFAR-100 test transforms."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.cifar100_mean, self.config.cifar100_std)
        ])


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def create_dataloader(dataset, config: DataConfig, is_train: bool = True) -> DataLoader:
        """Create a data loader with appropriate settings."""
        dataloader_args = {
            'batch_size': config.batch_size,
            'shuffle': config.shuffle if is_train else False,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory
        }
        
        # Adjust batch size for CPU
        if not torch.cuda.is_available():
            dataloader_args['batch_size'] = min(config.batch_size, 64)
            dataloader_args['num_workers'] = 0
        
        return DataLoader(dataset, **dataloader_args)


class CIFAR100DataManager:
    """
    Data manager class for CIFAR-100 dataset.
    Implements the Facade pattern to provide a simple interface for data operations.
    """
    
    def __init__(self, config: DataConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
        # CIFAR-100 class names
        # CIFAR-100 class names (100 classes)
        self.classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
        #self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Initialize transform strategies
        self.train_transform_strategy = CIFAR100TrainTransformStrategy(config)
        self.test_transform_strategy = CIFAR100TestTransformStrategy(config)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-100 dataset and create data loaders."""
        self.logger.info("Loading CIFAR-100 dataset...")
        
        # Create datasets
        self.train_dataset = datasets.CIFAR100(
            self.config.data_dir,
            train=True,
            download=True,
            transform=self.train_transform_strategy.get_transforms()
        )
        
        self.test_dataset = datasets.CIFAR100(
            self.config.data_dir,
            train=False,
            download=True,
            transform=self.test_transform_strategy.get_transforms()
        )
        
        # Create data loaders
        self.train_loader = DataLoaderFactory.create_dataloader(
            self.train_dataset, self.config, is_train=True
        )
        self.test_loader = DataLoaderFactory.create_dataloader(
            self.test_dataset, self.config, is_train=False
        )
        
        self.logger.info(f"CIFAR-100 dataset loaded successfully!")
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics for CIFAR-100."""
        self.logger.info("Computing CIFAR-100 data statistics...")
        
        # Get raw data for statistics
        raw_train_data = self.train_dataset.data
        float_data = torch.tensor(raw_train_data, dtype=torch.float32) / 255.0
        
        # Overall statistics
        overall_stats = {
            'shape': raw_train_data.shape,
            'size': raw_train_data.size,
            'min': torch.min(float_data).item(),
            'max': torch.max(float_data).item(),
            'mean': torch.mean(float_data).item(),
            'std': torch.std(float_data).item(),
            'var': torch.var(float_data).item()
        }
        
        # Channel-wise statistics
        channel_stats = {}
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            channel_data = float_data[:, :, :, i]
            channel_stats[channel.lower()] = {
                'min': torch.min(channel_data).item(),
                'max': torch.max(channel_data).item(),
                'mean': torch.mean(channel_data).item(),
                'median': torch.median(channel_data).item(),
                'std': torch.std(channel_data).item(),
                'var': torch.var(channel_data).item(),
                'range': (torch.max(channel_data) - torch.min(channel_data)).item()
            }
        
        stats = {
            'overall': overall_stats,
            'channels': channel_stats,
            'class_names': self.classes,
            'num_classes': len(self.classes)
        }
        
        # Log statistics
        self.logger.info("CIFAR-100 Data Statistics:")
        self.logger.info(f"  - Shape: {stats['overall']['shape']}")
        self.logger.info(f"  - Size: {stats['overall']['size']:,}")
        self.logger.info(f"  - Min: {stats['overall']['min']:.4f}")
        self.logger.info(f"  - Max: {stats['overall']['max']:.4f}")
        self.logger.info(f"  - Mean: {stats['overall']['mean']:.4f}")
        self.logger.info(f"  - Std: {stats['overall']['std']:.4f}")
        self.logger.info(f"  - Variance: {stats['overall']['var']:.4f}")
        
        self.logger.info("Channel-wise Statistics:")
        for channel, channel_stat in channel_stats.items():
            self.logger.info(f"  {channel.capitalize()} Channel:")
            self.logger.info(f"    - Mean: {channel_stat['mean']:.4f}")
            self.logger.info(f"    - Std: {channel_stat['std']:.4f}")
            self.logger.info(f"    - Min: {channel_stat['min']:.4f}")
            self.logger.info(f"    - Max: {channel_stat['max']:.4f}")
        
        return stats
    
    def get_input_size_from_dataloader(self) -> Tuple[int, int, int]:
        """Get input size dynamically from the first batch of data."""
        self.logger.info("Getting input size from CIFAR-100 data loader...")
        
        # Get a sample batch
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Extract input size (channels, height, width)
        input_size = images.shape[1:]  # Remove batch dimension
        input_size_tuple = tuple(input_size)
        
        self.logger.info(f"CIFAR-100 input size from data loader: {input_size_tuple}")
        return input_size_tuple
    
    def visualize_samples(self, num_samples: int = 16, save_path: Optional[str] = None) -> None:
        """Visualize sample images from the CIFAR-100 dataset."""
        self.logger.info(f"Visualizing {num_samples} CIFAR-100 sample images...")
        
        # Get a batch of data
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Create visualization with RGB channels
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('CIFAR-100 Sample Images', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            row = i // 4
            col = i % 4
            
            # Convert from (C, H, W) to (H, W, C) for display
            img = images[i].permute(1, 2, 0)
            
            # Denormalize for display
            img = img * torch.tensor(self.config.cifar100_std) + torch.tensor(self.config.cifar100_mean)
            img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img.numpy())
            axes[row, col].set_title(f'{self.classes[labels[i]]}', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"CIFAR-100 sample visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_rgb_channels(self, num_samples: int = 4, save_path: Optional[str] = None) -> None:
        """Visualize RGB channel analysis for CIFAR-100 samples."""
        self.logger.info(f"Visualizing RGB channel analysis for {num_samples} CIFAR-100 samples...")
        
        # Get a batch of data
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Create visualization showing original + RGB channels
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        fig.suptitle('CIFAR-100 RGB Channel Analysis', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            # Convert from (C, H, W) to (H, W, C) for display
            img = images[i].permute(1, 2, 0)
            
            # Denormalize for display
            img = img * torch.tensor(self.config.cifar100_std) + torch.tensor(self.config.cifar100_mean)
            img = torch.clamp(img, 0, 1)
            
            # Original RGB image
            axes[i, 0].imshow(img.numpy())
            axes[i, 0].set_title(f'Original: {self.classes[labels[i]]}', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Red channel
            axes[i, 1].imshow(img[:,:,0].numpy(), cmap='Reds')
            axes[i, 1].set_title('Red Channel', fontsize=10)
            axes[i, 1].axis('off')
            
            # Green channel
            axes[i, 2].imshow(img[:,:,1].numpy(), cmap='Greens')
            axes[i, 2].set_title('Green Channel', fontsize=10)
            axes[i, 2].axis('off')
            
            # Blue channel
            axes[i, 3].imshow(img[:,:,2].numpy(), cmap='Blues')
            axes[i, 3].set_title('Blue Channel', fontsize=10)
            axes[i, 3].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"CIFAR-100 RGB channel visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_channel_histograms(self, save_path: Optional[str] = None) -> None:
        """Plot histograms for each RGB channel."""
        self.logger.info("Plotting CIFAR-100 channel histograms...")
        
        # Get raw data for histogram analysis
        raw_train_data = self.train_dataset.data
        float_data = torch.tensor(raw_train_data, dtype=torch.float32) / 255.0
        
        fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(15, 5))
        
        # Calculate statistics for each channel
        red_data = float_data[:,:,:,0].ravel()
        green_data = float_data[:,:,:,1].ravel()
        blue_data = float_data[:,:,:,2].ravel()
        
        # Red channel
        axs[0].hist(red_data.numpy(), bins=255, color='red', alpha=0.7)
        red_stats = f'Mean: {torch.mean(red_data):.3f}\nMedian: {torch.median(red_data):.3f}\nStd: {torch.std(red_data):.3f}\nMin: {torch.min(red_data):.3f}\nMax: {torch.max(red_data):.3f}'
        axs[0].set_title('Red Channel', fontsize=12, fontweight='bold')
        axs[0].text(0.02, 0.98, red_stats, transform=axs[0].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axs[0].set_xlabel('Pixel Value')
        axs[0].set_ylabel('Frequency')
        
        # Green channel
        axs[1].hist(green_data.numpy(), bins=255, color='green', alpha=0.7)
        green_stats = f'Mean: {torch.mean(green_data):.3f}\nMedian: {torch.median(green_data):.3f}\nStd: {torch.std(green_data):.3f}\nMin: {torch.min(green_data):.3f}\nMax: {torch.max(green_data):.3f}'
        axs[1].set_title('Green Channel', fontsize=12, fontweight='bold')
        axs[1].text(0.02, 0.98, green_stats, transform=axs[1].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axs[1].set_xlabel('Pixel Value')
        
        # Blue channel
        axs[2].hist(blue_data.numpy(), bins=255, color='blue', alpha=0.7)
        blue_stats = f'Mean: {torch.mean(blue_data):.3f}\nMedian: {torch.median(blue_data):.3f}\nStd: {torch.std(blue_data):.3f}\nMin: {torch.min(blue_data):.3f}\nMax: {torch.max(blue_data):.3f}'
        axs[2].set_title('Blue Channel', fontsize=12, fontweight='bold')
        axs[2].text(0.02, 0.98, blue_stats, transform=axs[2].transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axs[2].set_xlabel('Pixel Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"CIFAR-100 channel histograms saved to: {save_path}")
        
        plt.show()
    
    def get_batch_info(self) -> Dict[str, Any]:
        """Get information about a sample batch."""
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        batch_info = {
            'batch_size': images.shape[0],
            'image_shape': images.shape[1:],
            'label_shape': labels.shape,
            'data_type': str(images.dtype),
            'device': str(images.device),
            'num_classes': len(self.classes),
            'class_names': self.classes
        }
        
        self.logger.info("CIFAR-100 Batch Information:")
        self.logger.info(f"  - Batch size: {batch_info['batch_size']}")
        self.logger.info(f"  - Image shape: {batch_info['image_shape']}")
        self.logger.info(f"  - Label shape: {batch_info['label_shape']}")
        self.logger.info(f"  - Data type: {batch_info['data_type']}")
        self.logger.info(f"  - Number of classes: {batch_info['num_classes']}")
        
        return batch_info
