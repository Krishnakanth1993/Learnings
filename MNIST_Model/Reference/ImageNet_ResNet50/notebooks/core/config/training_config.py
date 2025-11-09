"""
Training Configuration for ResNet50
Contains configuration classes for training.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for training ResNet50 model."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "SGD"
    scheduler: str = "StepLR"
    step_size: int = 30
    gamma: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: int = 10
    validation_frequency: int = 1
    mixed_precision: bool = False
    gradient_clipping: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'device': self.device,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_frequency': self.validation_frequency,
            'mixed_precision': self.mixed_precision,
            'gradient_clipping': self.gradient_clipping
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)