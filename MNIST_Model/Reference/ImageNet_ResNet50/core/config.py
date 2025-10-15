"""
Configuration Management for ImageNet ResNet-50 Training
Provides all configuration dataclasses following OOP principles.

Author: Krishnakanth
Date: 2025-10-13
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = '~/data/imagenet'
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    shuffle: bool = True
    persistent_workers: bool = True
    
    # ImageNet-1k specific normalization values
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # Input image size (progressive resizing support)
    input_size: int = 224  # Standard ImageNet size
    
    # Subset training support
    use_subset: bool = False
    subset_percentage: float = 0.2  # For phase 1: 20% stratified sampling
    subset_seed: int = 42  # For reproducible subset selection
    cache_subset_indices: bool = True  # Cache indices for consistency
    
    # Data augmentation parameters
    random_resized_crop_scale: Tuple[float, float] = (0.08, 1.0)
    random_horizontal_flip_p: float = 0.5
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    
    # Advanced augmentation (phase 2+)
    use_randaugment: bool = False
    randaugment_n: int = 2
    randaugment_m: int = 9
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_name: str = 'resnet50'
    input_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    num_classes: int = 1000  # ImageNet-1k
    dropout_rate: float = 0.0
    
    # ResNet-50 specific
    use_bottleneck: bool = True
    layers: Tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50 structure
    
    # Pretrained weights
    use_pretrained: bool = False
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False  # For transfer learning
    
    # Advanced features
    use_se_blocks: bool = False  # Squeeze-and-Excitation
    stochastic_depth_prob: float = 0.0  # Stochastic depth for regularization


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 100
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Gradient accumulation for memory efficiency
    gradient_accumulation_steps: int = 4  # Effective batch = 32*4 = 128
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0
    
    # Optimizer configuration
    optimizer_type: str = 'SGD'  # 'SGD', 'Adam', 'AdamW', 'RMSprop'
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    rmsprop_alpha: float = 0.99
    
    # Scheduler configuration
    scheduler_type: str = 'ReduceLROnPlateau'
    
    # OneCycleLR parameters
    onecycle_max_lr: float = 0.003
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 5.0
    onecycle_final_div_factor: float = 1000.0
    onecycle_anneal_strategy: str = 'cos'
    
    # ReduceLROnPlateau parameters
    plateau_mode: str = 'max'  # Based on accuracy
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    
    # CosineAnnealingLR parameters
    cosine_t_max: int = 30
    cosine_eta_min: float = 1e-6
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # EMA (Exponential Moving Average)
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False  # Set True for full reproducibility (slower)


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    log_dir: str = './logs'
    log_file: str = 'imagenet_training.log'
    log_level: str = 'INFO'
    
    # Model saving
    save_model: bool = True
    model_save_dir: str = './models'
    
    # Tensorboard logging
    use_tensorboard: bool = False
    tensorboard_dir: str = './runs'
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = 'imagenet-resnet50'
    wandb_entity: Optional[str] = None
    
    # Console output
    print_freq: int = 100  # Print every N batches
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    save_config: bool = True


@dataclass
class Config:
    """Main configuration class combining all configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        # Create all required directories
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.model_save_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        
        if self.logging.use_tensorboard:
            os.makedirs(self.logging.tensorboard_dir, exist_ok=True)
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.data.batch_size * self.training.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'logging': self.logging.__dict__
        }

