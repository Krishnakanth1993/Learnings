"""
MNIST Training Script - Complete Monolithic Implementation
A comprehensive, object-oriented Python script for training MNIST digit classification models.

This script combines all functionality into a single file while maintaining:
- Object-Oriented Design Principles
- Design Patterns (Singleton, Factory, Strategy, Builder, Facade, Template Method)
- Comprehensive logging and experiment tracking
- Flexible configuration management
- Data statistics and visualization
- Model architecture analysis
- Training progress monitoring

Author: AI Assistant
Date: 2024
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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from io import StringIO
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Import model classes from separate module
from model import ModelConfig, ModelBuilder, MNISTModel


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
    # MNIST specific normalization values
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)
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
    scheduler_step_size: int = 6
    scheduler_gamma: float = 0.1
    seed: int = 1


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = None  # Will be set to script directory
    log_file: str = 'mnist_training.log'
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
        self.logger = logging.getLogger('MNIST_Training')
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
        self.info("MNIST TRAINING EXPERIMENT STARTED")
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
        self.info(f"  - Momentum: {config.training.momentum}")
        self.info(f"  - Weight Decay: {config.training.weight_decay}")
        self.info(f"  - Scheduler Step Size: {config.training.scheduler_step_size}")
        self.info(f"  - Scheduler Gamma: {config.training.scheduler_gamma}")
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
                         test_loss: float, test_acc: float) -> None:
        """Log epoch results."""
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


# =============================================================================
# DATA LOADING SYSTEM (Factory & Strategy Patterns)
# =============================================================================

class TransformStrategy(ABC):
    """Abstract base class for transformation strategies."""
    
    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Get the composed transforms."""
        pass


class TrainTransformStrategy(TransformStrategy):
    """Strategy for training data transformations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_transforms(self) -> transforms.Compose:
        """Get training transforms with augmentation."""
        return transforms.Compose([
            transforms.RandomRotation(
                self.config.rotation_range, 
                fill=(self.config.fill_value,)
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std)
        ])


class TestTransformStrategy(TransformStrategy):
    """Strategy for test data transformations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_transforms(self) -> transforms.Compose:
        """Get test transforms without augmentation."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std)
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


class MNISTDataManager:
    """
    Data manager class for MNIST dataset.
    Implements the Facade pattern to provide a simple interface for data operations.
    """
    
    def __init__(self, config: DataConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
        # Initialize transform strategies
        self.train_transform_strategy = TrainTransformStrategy(config)
        self.test_transform_strategy = TestTransformStrategy(config)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load MNIST dataset and create data loaders."""
        self.logger.info("Loading MNIST dataset...")
        
        # Create datasets
        self.train_dataset = datasets.MNIST(
            self.config.data_dir,
            train=True,
            download=True,
            transform=self.train_transform_strategy.get_transforms()
        )
        
        self.test_dataset = datasets.MNIST(
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
        
        self.logger.info(f"Dataset loaded successfully!")
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Test samples: {len(self.test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        self.logger.info("Computing data statistics...")
        
        # Get raw data for statistics
        raw_train_data = self.train_dataset.data.float() / 255.0
        
        stats = {
            'shape': raw_train_data.shape,
            'size': raw_train_data.numel(),
            'min': torch.min(raw_train_data).item(),
            'max': torch.max(raw_train_data).item(),
            'mean': torch.mean(raw_train_data).item(),
            'std': torch.std(raw_train_data).item(),
            'var': torch.var(raw_train_data).item()
        }
        
        # Log statistics
        self.logger.info("Data Statistics:")
        self.logger.info(f"  - Shape: {stats['shape']}")
        self.logger.info(f"  - Size: {stats['size']:,}")
        self.logger.info(f"  - Min: {stats['min']:.4f}")
        self.logger.info(f"  - Max: {stats['max']:.4f}")
        self.logger.info(f"  - Mean: {stats['mean']:.4f}")
        self.logger.info(f"  - Std: {stats['std']:.4f}")
        self.logger.info(f"  - Variance: {stats['var']:.4f}")
        
        return stats
    
    def get_input_size_from_dataloader(self) -> Tuple[int, int, int]:
        """Get input size dynamically from the first batch of data."""
        self.logger.info("Getting input size from data loader...")
        
        # Get a sample batch
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Extract input size (channels, height, width)
        input_size = images.shape[1:]  # Remove batch dimension
        input_size_tuple = tuple(input_size)
        
        self.logger.info(f"Input size from data loader: {input_size_tuple}")
        return input_size_tuple
    
    def visualize_samples(self, num_samples: int = 60, save_path: Optional[str] = None) -> None:
        """Visualize sample images from the dataset."""
        self.logger.info(f"Visualizing {num_samples} sample images...")
        
        # Get a batch of data
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        
        # Create visualization
        fig, axes = plt.subplots(6, 10, figsize=(15, 9))
        fig.suptitle('MNIST Sample Images', fontsize=16)
        
        for i in range(min(num_samples, len(images))):
            row = i // 10
            col = i % 10
            axes[row, col].imshow(images[i].squeeze().numpy(), cmap='gray_r')
            axes[row, col].set_title(f'Label: {labels[i].item()}', fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Sample visualization saved to: {save_path}")
        
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
            'device': str(images.device)
        }
        
        self.logger.info("Batch Information:")
        self.logger.info(f"  - Batch size: {batch_info['batch_size']}")
        self.logger.info(f"  - Image shape: {batch_info['image_shape']}")
        self.logger.info(f"  - Label shape: {batch_info['label_shape']}")
        self.logger.info(f"  - Data type: {batch_info['data_type']}")
        
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
    
    def add_epoch_metrics(self, train_loss: float, train_acc: float, 
                         test_loss: float, test_acc: float) -> None:
        """Add metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
    
    def get_best_accuracy(self) -> float:
        """Get the best test accuracy achieved."""
        return max(self.test_accuracies) if self.test_accuracies else 0.0
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        if not self.test_accuracies:
            return {}
        
        return {
            'final_train_loss': self.train_losses[-1],
            'final_test_loss': self.test_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_test_accuracy': self.test_accuracies[-1],
            'best_test_accuracy': self.get_best_accuracy()
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


class MNISTTrainer(BaseTrainer):
    """
    MNIST model trainer.
    Implements the Template Method pattern for training workflow.
    """
    
    def __init__(self, config: TrainingConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.metrics = TrainingMetrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        self.logger.info(f"Using device: {self.device}")
    
    def train_epoch(self, model, device, train_loader, optimizer, epoch: int) -> Tuple[float, float]:
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
        Complete training process.
        Template method that defines the training workflow.
        """
        self.logger.info("Starting training process...")
        
        # Setup optimizer and scheduler
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = StepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, self.device, train_loader, optimizer, epoch + 1
            )
            
            # Test
            test_loss, test_acc = self.test_epoch(model, self.device, test_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Store metrics
            self.metrics.add_epoch_metrics(train_loss, train_acc, test_loss, test_acc)
            
            # Log results
            self.logger.log_epoch_results(
                epoch + 1, train_loss, train_acc, test_loss, test_acc
            )
        
        # Log final results
        final_metrics = self.metrics.get_final_metrics()
        self.logger.info("Training completed!")
        self.logger.info(f"Final Results: {final_metrics}")
        
        return self.metrics
    
    def plot_training_curves(self, save_path: str = None) -> None:
        """Plot training curves."""
        if not self.metrics.train_losses:
            self.logger.warning("No training data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
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

class MNISTTrainingFacade:
    """
    Facade class that provides a simple interface for the entire MNIST training process.
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
        self.data_manager = MNISTDataManager(self.config.data, self.logger)
        train_loader, test_loader = self.data_manager.load_data()
        
        # Get and log data statistics
        self.data_manager.get_data_statistics()
        self.data_manager.get_batch_info()
        
        return train_loader, test_loader
    
    def setup_model(self, input_size: Tuple[int, int, int]) -> MNISTModel:
        """Setup the model with dynamic input size."""
        self.logger.info("Setting up model...")
        builder = ModelBuilder()
        self.model = builder.build_mnist_model(self.config.model)
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        # Get and log model summary with dynamic input size
        model_info = self.model.get_model_summary(input_size, self.logger)
        
        return self.model
    
    def setup_trainer(self) -> MNISTTrainer:
        """Setup the trainer."""
        self.logger.info("Setting up trainer...")
        self.trainer = MNISTTrainer(self.config.training, self.logger)
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
                f"mnist_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            )
            self.trainer.save_model(self.model, model_path)
        
        return metrics.get_final_metrics()
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete MNIST training pipeline."""
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
    Main function to run the MNIST training.
    Demonstrates usage of the complete training system.
    """
    print("=" * 60)
    print("MNIST TRAINING SCRIPT - MONOLITHIC IMPLEMENTATION")
    print("=" * 60)
    
    # Create configuration
    config = Config()
    
    # You can customize the configuration here
    config.training.epochs = 20
    config.training.learning_rate = 0.01

    config.training.momentum = 0.9
    config.data.batch_size = 64
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
    training_facade = MNISTTrainingFacade(config)
    
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