"""
CIFAR-100 Training Script - Complete Monolithic Implementation
A comprehensive, object-oriented Python script for training CIFAR-100 classification models.

This script combines all functionality into a single file while maintaining:
- Object-Oriented Design Principles
- Design Patterns (Singleton, Factory, Strategy, Builder, Facade, Template Method)
- Comprehensive logging and experiment tracking
- Flexible configuration management
- Data statistics and visualization
- Model architecture analysis
- Training progress monitoring

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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau,OneCycleLR
from tqdm import tqdm
from io import StringIO
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Import model classes from separate module
from model import ModelConfig, ModelBuilder, CIFAR100ResNet34


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

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
    scheduler_type: str = 'StepLR'  # Options: 'StepLR', 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau', 'OneCycleLR'
    cosine_t_max: int = 20
    exponential_gamma: float = 0.95
    plateau_mode: str = 'min'
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    
    # OneCycleLR configuration
    onecycle_max_lr: float = 0.1  # Maximum learning rate for OneCycleLR
    onecycle_pct_start: float = 0.3  # Percentage of cycle spent increasing LR
    onecycle_div_factor: float = 25.0  # Initial LR = max_lr / div_factor
    onecycle_final_div_factor: float = 10000.0  # Final LR = max_lr / final_div_factor
    onecycle_anneal_strategy: str = 'cos'  # 'cos' or 'linear'


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
    model: ModelConfig = field(default_factory=ModelConfig)
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
        elif config.training.scheduler_type == 'OneCycleLR':
            self.info(f"  - Max LR: {config.training.onecycle_max_lr}")
            self.info(f"  - Pct Start: {config.training.onecycle_pct_start}")
            self.info(f"  - Div Factor: {config.training.onecycle_div_factor}")
            self.info(f"  - Final Div Factor: {config.training.onecycle_final_div_factor}")
            self.info(f"  - Anneal Strategy: {config.training.onecycle_anneal_strategy}")
            
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
    def get_transforms(self):
        """Get the composed transforms."""
        pass
    
    @abstractmethod
    def __call__(self, image):
        """Apply transforms to an image."""
        pass


class AlbumentationsTransformStrategy(TransformStrategy):
    """Strategy for Albumentations transformations."""
    
    def __init__(self, transform_pipeline):
        """
        Initialize with an Albumentations Compose pipeline.
        
        Args:
            transform_pipeline: A.Compose object with transformations
        """
        self.transform_pipeline = transform_pipeline
    
    def get_transforms(self):
        """Get the Albumentations transform pipeline."""
        return self.transform_pipeline
    
    def __call__(self, image):
        """
        Apply Albumentations transforms to an image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Transformed tensor
        """
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Apply Albumentations transforms
        transformed = self.transform_pipeline(image=image)
        return transformed['image']


class CIFAR100TransformStrategy(AlbumentationsTransformStrategy):
    """Strategy for CIFAR-100 basic transformations (no augmentation)."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Create Albumentations pipeline for basic transforms
        transform_pipeline = A.Compose([
            A.Normalize(
                mean=self.config.cifar100_mean,
                std=self.config.cifar100_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        super().__init__(transform_pipeline)


class CIFAR100TrainTransformStrategy(AlbumentationsTransformStrategy):
    """Strategy for CIFAR-100 training data transformations with Albumentations augmentation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Create Albumentations pipeline with augmentations
        transform_pipeline = A.Compose([
            # Augmentations FIRST (work on 0-255 numpy arrays)
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,  
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=tuple([int(x * 255) for x in self.config.cifar100_mean]),  # CIFAR-100 mean values (0-255 scale)
                mask_fill_value=None,
                p=0.5
            ),
            # Normalize AFTER augmentation
            A.Normalize(
                mean=self.config.cifar100_mean,
                std=self.config.cifar100_std,
                max_pixel_value=255.0
            ),
            # Convert to tensor LAST
            ToTensorV2()
        ])
        
        super().__init__(transform_pipeline)


class CIFAR100TestTransformStrategy(AlbumentationsTransformStrategy):
    """Strategy for CIFAR-100 test data transformations (no augmentation)."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Create Albumentations pipeline for test (no augmentation)
        transform_pipeline = A.Compose([
            A.Normalize(
                mean=self.config.cifar100_mean,
                std=self.config.cifar100_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        super().__init__(transform_pipeline)


# PyTorch Transform Strategies (Alternative - if you want to keep PyTorch option)
class PyTorchTransformStrategy(TransformStrategy):
    """Strategy for PyTorch native transformations."""
    
    def __init__(self, transform_pipeline):
        """
        Initialize with a torchvision.transforms.Compose pipeline.
        
        Args:
            transform_pipeline: transforms.Compose object
        """
        self.transform_pipeline = transform_pipeline
    
    def get_transforms(self):
        """Get the PyTorch transform pipeline."""
        return self.transform_pipeline
    
    def __call__(self, image):
        """Apply PyTorch transforms to an image."""
        return self.transform_pipeline(image)


class CIFAR100PyTorchTrainTransformStrategy(PyTorchTransformStrategy):
    """Alternative PyTorch-only training transforms (if Albumentations not desired)."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        transform_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.cifar100_mean,
                std=self.config.cifar100_std
            ),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            )
        ])
        
        super().__init__(transform_pipeline)


class CIFAR100PyTorchTestTransformStrategy(PyTorchTransformStrategy):
    """Alternative PyTorch-only test transforms."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.cifar100_mean,
                std=self.config.cifar100_std
            )
        ])
        
        super().__init__(transform_pipeline)


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
    
    def __init__(self, config: DataConfig, logger: Logger, use_albumentations: bool = True):
        self.config = config
        self.logger = logger
        self.use_albumentations = use_albumentations
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
        # CIFAR-100 class names
        self.classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
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
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    
        # Initialize transform strategies based on choice
        if use_albumentations:
            self.logger.info("Using Albumentations for data augmentation")
            self.train_transform_strategy = CIFAR100TrainTransformStrategy(config)
            self.test_transform_strategy = CIFAR100TestTransformStrategy(config)
        else:
            self.logger.info("Using PyTorch native transforms for data augmentation")
            self.train_transform_strategy = CIFAR100PyTorchTrainTransformStrategy(config)
            self.test_transform_strategy = CIFAR100PyTorchTestTransformStrategy(config)

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-100 dataset and create data loaders."""
        self.logger.info("Loading CIFAR-100 dataset...")
        
        # Create datasets with transform strategies
        self.train_dataset = datasets.CIFAR100(
            self.config.data_dir,
            train=True,
            download=True,
            transform=self.train_transform_strategy  # Pass the strategy directly
        )
        
        self.test_dataset = datasets.CIFAR100(
            self.config.data_dir,
            train=False,
            download=True,
            transform=self.test_transform_strategy  # Pass the strategy directly
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
        self.logger.info(f"Augmentation library: {'Albumentations' if self.use_albumentations else 'PyTorch'}")
        
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

# =============================================================================
# TRAINING SYSTEM (Template Method Pattern)
# =============================================================================

class TrainingMetrics:
    """Class to track training metrics."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_accuracies: List[float] = []
        self.accuracy_differences: List[float] = []  # Track accuracy differences
        self.overfitting_epochs: int = 0  # Count consecutive overfitting epochs
    
    def add_epoch_metrics(self, train_loss: float, train_acc: float, 
                         test_loss: float, test_acc: float) -> None:
        """Add metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        # Calculate accuracy difference
        acc_diff = train_acc - test_acc
        self.accuracy_differences.append(acc_diff)
        
        # Check for overfitting (train_acc - test_acc > 15%)
        if acc_diff > 15.0:
            self.overfitting_epochs += 1
        else:
            self.overfitting_epochs = 0  # Reset counter if not overfitting
    
    def should_stop_training(self, patience: int = 10) -> bool:
        """Check if training should stop due to overfitting."""
        return self.overfitting_epochs >= patience
    
    def get_current_accuracy_difference(self) -> float:
        """Get the current accuracy difference."""
        return self.accuracy_differences[-1] if self.accuracy_differences else 0.0
    
    def get_overfitting_info(self) -> Dict[str, Any]:
        """Get overfitting information."""
        return {
            'current_difference': self.get_current_accuracy_difference(),
            'overfitting_epochs': self.overfitting_epochs,
            'should_stop': self.should_stop_training(),
            'max_difference': max(self.accuracy_differences) if self.accuracy_differences else 0.0,
            'avg_difference': sum(self.accuracy_differences) / len(self.accuracy_differences) if self.accuracy_differences else 0.0
        }
    
    def get_best_accuracy(self) -> float:
        """Get the best test accuracy achieved."""
        return max(self.test_accuracies) if self.test_accuracies else 0.0
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        if not self.test_accuracies:
            return {}
        
        overfitting_info = self.get_overfitting_info()
        
        return {
            'final_train_loss': self.train_losses[-1],
            'final_test_loss': self.test_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_test_accuracy': self.test_accuracies[-1],
            'best_test_accuracy': self.get_best_accuracy(),
            'final_accuracy_difference': self.get_current_accuracy_difference(),
            'max_accuracy_difference': overfitting_info['max_difference'],
            'avg_accuracy_difference': overfitting_info['avg_difference'],
            'overfitting_epochs': overfitting_info['overfitting_epochs'],
            'stopped_due_to_overfitting': overfitting_info['should_stop']
        }


class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    @abstractmethod
    def train_epoch(self, model, device, train_loader, optimizer, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def test_epoch(self, model, device, test_loader) -> Tuple[float, float]:
        """Test for one epoch."""
        pass


class CIFAR100Trainer(BaseTrainer):
    """
    CIFAR-100 model trainer.
    Implements the Template Method pattern for training workflow.
    """
    
    def __init__(self, config: TrainingConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.metrics = TrainingMetrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Early stopping configuration
        self.early_stopping_patience = 10  # Stop if overfitting for 10 epochs
        self.overfitting_threshold = 15.0  # 15% difference threshold
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Early stopping: Stop if train_acc - test_acc > {self.overfitting_threshold}% for {self.early_stopping_patience} epochs")
    
    def create_optimizer(self, model) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type == 'SGD':
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
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
        elif self.config.optimizer_type == 'RMSprop':
            return optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                alpha=self.config.rmsprop_alpha,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
    
    def create_scheduler(self, optimizer, steps_per_epoch: int = None) -> Any:
        """Create scheduler based on configuration."""
        if self.config.scheduler_type == 'StepLR':
            return StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler_type == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.cosine_t_max,
                eta_min=0
            )
        elif self.config.scheduler_type == 'ExponentialLR':
            return ExponentialLR(
                optimizer,
                gamma=self.config.exponential_gamma
            )
        elif self.config.scheduler_type == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(
                optimizer,
                mode=self.config.plateau_mode,
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                threshold=self.config.plateau_threshold
            )
        elif self.config.scheduler_type == 'OneCycleLR':
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires steps_per_epoch parameter")
            
            return OneCycleLR(
                optimizer,
                max_lr=self.config.onecycle_max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=self.config.epochs,
                pct_start=self.config.onecycle_pct_start,
                div_factor=self.config.onecycle_div_factor,
                final_div_factor=self.config.onecycle_final_div_factor,
                anneal_strategy=self.config.onecycle_anneal_strategy
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.config.scheduler_type}")
    
    def train_epoch(self, model, device, train_loader, optimizer, epoch: int, scheduler=None) -> Tuple[float, float]:
        """Train the model for one epoch."""
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        correct = 0
        processed = 0
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update OneCycleLR scheduler per batch
            if scheduler is not None and self.config.scheduler_type == 'OneCycleLR':
                scheduler.step()
            
            # Update metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch} - Loss: {loss.item():.4f}, '
                f'Accuracy: {100*correct/processed:.2f}%'
            )
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / processed
        
        return avg_loss, accuracy
    
    def test_epoch(self, model, device, test_loader) -> Tuple[float, float]:
        """Test the model for one epoch."""
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        return test_loss, accuracy
    
    def train(self, model, train_loader, test_loader) -> TrainingMetrics:
        """
        Complete training process with early stopping.
        Template method that defines the training workflow.
        """
        self.logger.info("Starting training process...")
        
        # Setup optimizer and scheduler
        optimizer = self.create_optimizer(model)
        
        # Calculate steps per epoch for OneCycleLR
        steps_per_epoch = len(train_loader)
        scheduler = self.create_scheduler(optimizer, steps_per_epoch=steps_per_epoch)
        
        self.logger.info(f"Using optimizer: {self.config.optimizer_type}")
        self.logger.info(f"Using scheduler: {self.config.scheduler_type}")
        
        # Log optimizer details
        self.logger.info("Optimizer Configuration:")
        if self.config.optimizer_type == 'SGD':
            self.logger.info(f"  - Learning Rate: {self.config.learning_rate}")
            self.logger.info(f"  - Momentum: {self.config.momentum}")
            self.logger.info(f"  - Weight Decay: {self.config.weight_decay}")
        elif self.config.optimizer_type in ['Adam', 'AdamW']:
            self.logger.info(f"  - Learning Rate: {self.config.learning_rate}")
            self.logger.info(f"  - Betas: {self.config.adam_betas}")
            self.logger.info(f"  - Eps: {self.config.adam_eps}")
            self.logger.info(f"  - Weight Decay: {self.config.weight_decay}")
        elif self.config.optimizer_type == 'RMSprop':
            self.logger.info(f"  - Learning Rate: {self.config.learning_rate}")
            self.logger.info(f"  - Alpha: {self.config.rmsprop_alpha}")
            self.logger.info(f"  - Weight Decay: {self.config.weight_decay}")
        
        # Log scheduler details
        self.logger.info("Scheduler Configuration:")
        if self.config.scheduler_type == 'StepLR':
            self.logger.info(f"  - Step Size: {self.config.scheduler_step_size}")
            self.logger.info(f"  - Gamma: {self.config.scheduler_gamma}")
        elif self.config.scheduler_type == 'CosineAnnealingLR':
            self.logger.info(f"  - T Max: {self.config.cosine_t_max}")
            self.logger.info(f"  - Eta Min: 0")
        elif self.config.scheduler_type == 'ExponentialLR':
            self.logger.info(f"  - Gamma: {self.config.exponential_gamma}")
        elif self.config.scheduler_type == 'ReduceLROnPlateau':
            self.logger.info(f"  - Mode: {self.config.plateau_mode}")
            self.logger.info(f"  - Factor: {self.config.plateau_factor}")
            self.logger.info(f"  - Patience: {self.config.plateau_patience}")
            self.logger.info(f"  - Threshold: {self.config.plateau_threshold}")
        elif self.config.scheduler_type == 'OneCycleLR':
            self.logger.info(f"  - Max LR: {self.config.onecycle_max_lr}")
            self.logger.info(f"  - Steps per Epoch: {steps_per_epoch}")
            self.logger.info(f"  - Pct Start: {self.config.onecycle_pct_start}")
            self.logger.info(f"  - Div Factor: {self.config.onecycle_div_factor}")
            self.logger.info(f"  - Final Div Factor: {self.config.onecycle_final_div_factor}")
            self.logger.info(f"  - Anneal Strategy: {self.config.onecycle_anneal_strategy}")
        
        # Training loop with early stopping
        for epoch in range(self.config.epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, self.device, train_loader, optimizer, epoch + 1, scheduler
            )
            
            # Test
            test_loss, test_acc = self.test_epoch(model, self.device, test_loader)
            
            # Update scheduler (except OneCycleLR which updates per batch)
            if self.config.scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(test_loss)  # ReduceLROnPlateau needs a metric
            elif self.config.scheduler_type != 'OneCycleLR':
                scheduler.step()
            # OneCycleLR is stepped inside train_epoch (per batch)
            
            # Store metrics
            self.metrics.add_epoch_metrics(train_loss, train_acc, test_loss, test_acc)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Get accuracy difference and overfitting info
            acc_diff = self.metrics.get_current_accuracy_difference()
            overfitting_info = self.metrics.get_overfitting_info()
            
            # Log results with accuracy difference
            self.logger.log_epoch_results_with_difference(
                epoch + 1, train_loss, train_acc, test_loss, test_acc, 
                current_lr, acc_diff, overfitting_info['overfitting_epochs']
            )
        
        # Log final results
        final_metrics = self.metrics.get_final_metrics()
        self.logger.info("Training completed!")
        self.logger.info(f"Final Results: {final_metrics}")
        
        # Log overfitting summary
        overfitting_info = self.metrics.get_overfitting_info()
        self.logger.info("=" * 50)
        self.logger.info("OVERFITTING ANALYSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"Final accuracy difference: {overfitting_info['current_difference']:.2f}%")
        self.logger.info(f"Maximum accuracy difference: {overfitting_info['max_difference']:.2f}%")
        self.logger.info(f"Average accuracy difference: {overfitting_info['avg_difference']:.2f}%")
        self.logger.info(f"Consecutive overfitting epochs: {overfitting_info['overfitting_epochs']}")
        self.logger.info(f"Stopped due to overfitting: {overfitting_info['should_stop']}")
        self.logger.info("=" * 50)
        
        return self.metrics


    def plot_training_curves(self, save_path: str = None) -> None:
        """Plot training curves with accuracy difference."""
        if not self.metrics.train_losses:
            self.logger.warning("No training data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress with Overfitting Analysis', fontsize=16)
        
        epochs = range(1, len(self.metrics.train_losses) + 1)
        
        # Training Loss
        axes[0, 0].plot(epochs, self.metrics.train_losses, 'b-', label='Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Training Accuracy
        axes[1, 0].plot(epochs, self.metrics.train_accuracies, 'b-', label='Train Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].grid(True)
        
        # Test Loss
        axes[0, 1].plot(epochs, self.metrics.test_losses, 'r-', label='Test Loss')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Test Accuracy
        axes[1, 1].plot(epochs, self.metrics.test_accuracies, 'r-', label='Test Accuracy')
        axes[1, 1].set_title('Test Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True)
        
        # Accuracy Difference
        axes[0, 2].plot(epochs, self.metrics.accuracy_differences, 'g-', label='Train - Test')
        axes[0, 2].axhline(y=self.overfitting_threshold, color='r', linestyle='--', 
                          label=f'Overfitting Threshold ({self.overfitting_threshold}%)')
        axes[0, 2].set_title('Accuracy Difference (Train - Test)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy Difference (%)')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        
        # Combined Accuracy Plot
        axes[1, 2].plot(epochs, self.metrics.train_accuracies, 'b-', label='Train Accuracy')
        axes[1, 2].plot(epochs, self.metrics.test_accuracies, 'r-', label='Test Accuracy')
        axes[1, 2].set_title('Train vs Test Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy (%)')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training curves saved to: {save_path}")
        
        plt.show()

    def save_model(self, model, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'metrics': self.metrics.get_final_metrics()
        }, filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, model, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from: {filepath}")


# =============================================================================
# MAIN TRAINING FACADE (Facade Pattern)
# =============================================================================

class TrainingFacade:
    """
    Facade class that provides a simple interface for the entire CIFAR-10 training process.
    Implements the Facade pattern to hide the complexity of the training system.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = Logger(self.config.logging)
        self.data_manager = None
        self.model = None
        self.trainer = None
    
    def setup_data(self) -> tuple:
        """Setup data loading and get data loaders."""
        self.logger.info("Setting up data...")
        self.data_manager = CIFAR100DataManager(self.config.data, self.logger)
        train_loader, test_loader = self.data_manager.load_data()
        
        # Get and log data statistics
        self.data_manager.get_data_statistics()
        self.data_manager.get_batch_info()
        
        return train_loader, test_loader
    
    def setup_model(self, input_size: Tuple[int, int, int]) -> CIFAR100ResNet34:
        """Setup the model with dynamic input size."""
        self.logger.info("Setting up model...")
        builder = ModelBuilder()
        self.model = builder.build_resnet34(self.config.model).build()
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        # Get and log model summary with dynamic input size
        model_info = self.model.get_model_summary(input_size, self.logger)
        
        return self.model
    
    def setup_trainer(self) -> CIFAR100Trainer:
        """Setup the trainer."""
        self.logger.info("Setting up trainer...")
        self.trainer = CIFAR100Trainer(self.config.training, self.logger)
        return self.trainer
    
    def run_training(self, train_loader, test_loader) -> dict:
        """Run the complete training process."""
        self.logger.info("Starting training process...")
        
        # Train the model
        metrics = self.trainer.train(self.model, train_loader, test_loader)
        
        # Plot training curves
        curves_path = os.path.join(
            self.config.logging.log_dir,
            f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        self.trainer.plot_training_curves(curves_path)
        
        # Save model if configured
        if self.config.logging.save_model:
            model_path = os.path.join(
                self.config.logging.model_save_dir,
                f"cifar100_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            )
            self.trainer.save_model(self.model, model_path)
        
        return metrics.get_final_metrics()
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete CIFAR-100 training pipeline."""
        try:
            # Log experiment start
            self.logger.log_experiment_start(self.config)
            
            # Setup components
            train_loader, test_loader = self.setup_data()
            
            # Get input size dynamically from data loader
            input_size = self.data_manager.get_input_size_from_dataloader()
            
            # Setup model with dynamic input size
            model = self.setup_model(input_size)
            trainer = self.setup_trainer()
            
            # Run training
            final_metrics = self.run_training(train_loader, test_loader)
            
            # Log completion
            self.logger.info("=" * 50)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 50)
            self.logger.info(f"Final Metrics: {final_metrics}")
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the CIFAR-100 training.
    Demonstrates usage of the complete training system.
    """
    print("=" * 60)
    print("CIFAR-100 TRAINING SCRIPT - MONOLITHIC IMPLEMENTATION")
    print("=" * 60)
    
    # Create configuration
    config = Config()
    
    # You can customize the configuration here
    config.training.epochs = 100
    config.training.learning_rate = 0.001
    config.training.momentum = 0.9
    config.data.batch_size = 128
    #config.training.scheduler_step_size = 20
    config.training.optimizer_type = 'Adam'
    config.training.weight_decay = 0.0001
    config.model.dropout_rate = 0.2


    # config.training.scheduler_type = 'OneCycleLR'
    # config.training.onecycle_max_lr = 2.51e-02
    # config.training.onecycle_pct_start = 0.3
    # config.training.onecycle_div_factor = 25.0
    # config.training.onecycle_final_div_factor = 10000.0
    # config.training.onecycle_anneal_strategy = 'cos'

    config.training.scheduler_type = 'ReduceLROnPlateau'
    config.training.plateau_mode = 'min'
    config.training.plateau_factor = 0.5
    config.training.plateau_patience = 10
    config.training.plateau_threshold = 1e-4
    config.training.weight_decay = 0.0001
    config.logging.log_level = 'DEBUG'
    
    print(f"Configuration:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Learning Rate: {config.training.learning_rate}")
    print(f"  - Batch Size: {config.data.batch_size}")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Log Directory: {config.logging.log_dir}")
    print(f"  - Model Save Directory: {config.logging.model_save_dir}")
    print("=" * 60)
    
    # Create and run the training facade
    training_facade = TrainingFacade(config)
    
    # Log the updated configuration after all updates
    training_facade.logger.log_updated_config(config)
    final_metrics = training_facade.run_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Test Accuracy: {final_metrics['final_test_accuracy']:.2f}%")
    print(f"Best Test Accuracy: {final_metrics['best_test_accuracy']:.2f}%")
    print(f"Final Train Accuracy: {final_metrics['final_train_accuracy']:.2f}%")
    print("=" * 60)
    print("Check the logs/ directory for detailed logs and visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()