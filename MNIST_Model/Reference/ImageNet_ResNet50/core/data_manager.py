"""
Data Management for ImageNet Dataset
Supports stratified sampling, progressive resizing, and Albumentations.

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from abc import ABC, abstractmethod
import os
import pickle
from PIL import Image

from .config import DataConfig
from .logger import Logger


class StratifiedSubsetSampler:
    """
    Creates stratified subset of dataset (same percentage from each class).
    Ensures balanced class distribution in subset.
    """
    
    def __init__(self, dataset: Dataset, percentage: float, seed: int = 42,
                 cache_path: Optional[str] = None):
        """
        Initialize stratified sampler.
        
        Args:
            dataset: PyTorch dataset with targets attribute
            percentage: Fraction of data to sample (0.0-1.0)
            seed: Random seed for reproducibility
            cache_path: Path to cache indices
        """
        self.dataset = dataset
        self.percentage = percentage
        self.seed = seed
        self.cache_path = cache_path
        self.indices = None
    
    def get_subset_indices(self) -> List[int]:
        """Get stratified subset indices."""
        # Try loading cached indices
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if (cached_data['percentage'] == self.percentage and 
                    cached_data['seed'] == self.seed and
                    cached_data['dataset_size'] == len(self.dataset)):
                    return cached_data['indices']
        
        # Generate new stratified indices
        np.random.seed(self.seed)
        
        # Get all targets
        if hasattr(self.dataset, 'targets'):
            targets = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            targets = np.array(self.dataset.labels)
        else:
            raise AttributeError("Dataset must have 'targets' or 'labels' attribute")
        
        # Get unique classes
        classes = np.unique(targets)
        indices = []
        
        # Sample stratified
        for cls in classes:
            cls_indices = np.where(targets == cls)[0]
            n_samples = int(len(cls_indices) * self.percentage)
            sampled = np.random.choice(cls_indices, size=n_samples, replace=False)
            indices.extend(sampled.tolist())
        
        indices = sorted(indices)
        
        # Cache indices
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            cache_data = {
                'indices': indices,
                'percentage': self.percentage,
                'seed': self.seed,
                'dataset_size': len(self.dataset)
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        self.indices = indices
        return indices
    
    def create_subset(self) -> Subset:
        """Create stratified subset of dataset."""
        indices = self.get_subset_indices()
        return Subset(self.dataset, indices)


class TransformStrategy(ABC):
    """Abstract base class for transformation strategies."""
    
    @abstractmethod
    def __call__(self, image):
        """Apply transforms to an image."""
        pass


class AlbumentationsTransform(TransformStrategy):
    """Albumentations transformation strategy."""
    
    def __init__(self, transform_pipeline):
        self.transform_pipeline = transform_pipeline
    
    def __call__(self, image):
        """Apply Albumentations transforms."""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        transformed = self.transform_pipeline(image=image)
        return transformed['image']


class ImageNetDataManager:
    """
    Data manager for ImageNet dataset with stratified sampling support.
    Implements Facade pattern for data operations.
    """
    
    def __init__(self, config: DataConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.subset_indices = None
    
    def _get_train_transforms(self) -> AlbumentationsTransform:
        """Get training data transforms with Albumentations."""
        transforms_list = [
            A.RandomResizedCrop(
                (self.config.input_size,self.config.input_size),
                scale=self.config.random_resized_crop_scale,
                p=1.0
            ),
            A.HorizontalFlip(p=self.config.random_horizontal_flip_p),
            A.ColorJitter(
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
                hue=self.config.color_jitter_hue,
                p=0.8
            ),
        ]
        
        # Add RandAugment if enabled
        if self.config.use_randaugment:
            # Enhanced augmentations similar to RandAugment
            self.logger.info(f"RandAugment enabled: N={self.config.randaugment_n}, M={self.config.randaugment_m}")
            transforms_list.extend([
                A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                ], p=0.3),
            ])
        
        # Add normalization and tensor conversion
        transforms_list.extend([
            A.Normalize(
                mean=self.config.imagenet_mean,
                std=self.config.imagenet_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        transform_pipeline = A.Compose(transforms_list)
        
        return AlbumentationsTransform(transform_pipeline)
    
    def _get_val_transforms(self) -> AlbumentationsTransform:
        """Get validation data transforms (no augmentation)."""
        transform_pipeline = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=self.config.input_size, width=self.config.input_size),
            A.Normalize(
                mean=self.config.imagenet_mean,
                std=self.config.imagenet_std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        return AlbumentationsTransform(transform_pipeline)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load ImageNet dataset and create data loaders.
        
        Returns:
            train_loader, val_loader
        """
        self.logger.info("Loading ImageNet dataset...")
        
        # Check if ImageNet data exists
        train_dir = os.path.join(self.config.data_dir, 'train')
        val_dir = os.path.join(self.config.data_dir, 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            error_msg = (
                f"ImageNet data not found at {self.config.data_dir}\n"
                f"Please download ImageNet-1k dataset manually and extract to:\n"
                f"  - {train_dir} (training set)\n"
                f"  - {val_dir} (validation set)\n"
                f"See README.md for download instructions."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Create datasets
        self.logger.info(f"Train directory: {train_dir}")
        self.logger.info(f"Val directory: {val_dir}")
        
        train_dataset_full = datasets.ImageFolder(
            train_dir,
            transform=self._get_train_transforms()
        )
        
        self.val_dataset = datasets.ImageFolder(
            val_dir,
            transform=self._get_val_transforms()
        )
        
        # Apply stratified sampling if configured
        if self.config.use_subset:
            self.logger.info(f"Creating stratified {self.config.subset_percentage*100}% subset...")
            
            cache_path = None
            if self.config.cache_subset_indices:
                cache_path = os.path.join(
                    self.config.data_dir,
                    f'subset_{int(self.config.subset_percentage*100)}pct_seed{self.config.subset_seed}.pkl'
                )
            
            sampler = StratifiedSubsetSampler(
                train_dataset_full,
                self.config.subset_percentage,
                self.config.subset_seed,
                cache_path
            )
            
            self.train_dataset = sampler.create_subset()
            self.subset_indices = sampler.indices
            
            self.logger.info(f"Subset created: {len(self.train_dataset)} / {len(train_dataset_full)} samples")
        else:
            self.train_dataset = train_dataset_full
            self.logger.info(f"Using full training set: {len(self.train_dataset)} samples")
        
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False
        )
        
        self.logger.info("ImageNet data loaders created successfully!")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Val batches: {len(self.val_loader)}")
        
        return self.train_loader, self.val_loader
    
    def get_class_names(self) -> List[str]:
        """Get ImageNet class names."""
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        # Get class to idx mapping
        if hasattr(self.train_dataset, 'dataset'):
            # It's a Subset
            class_to_idx = self.train_dataset.dataset.class_to_idx
        else:
            class_to_idx = self.train_dataset.class_to_idx
        
        # Sort by index to get ordered class names
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        return class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes in dataset."""
        class_names = self.get_class_names()
        return len(class_names)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        self.logger.info("Computing dataset statistics...")
        
        stats = {
            'num_train_samples': len(self.train_dataset),
            'num_val_samples': len(self.val_dataset),
            'num_classes': self.get_num_classes(),
            'batch_size': self.config.batch_size,
            'num_train_batches': len(self.train_loader),
            'num_val_batches': len(self.val_loader),
            'input_size': self.config.input_size,
            'using_subset': self.config.use_subset,
        }
        
        if self.config.use_subset:
            stats['subset_percentage'] = self.config.subset_percentage
            stats['subset_samples'] = len(self.train_dataset)
        
        # Log statistics
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"  Train samples: {stats['num_train_samples']:,}")
        self.logger.info(f"  Val samples: {stats['num_val_samples']:,}")
        self.logger.info(f"  Num classes: {stats['num_classes']}")
        self.logger.info(f"  Batch size: {stats['batch_size']}")
        self.logger.info(f"  Input size: {stats['input_size']}x{stats['input_size']}")
        
        if self.config.use_subset:
            self.logger.info(f"  Subset: {stats['subset_percentage']*100}% stratified")
        
        return stats
    
    def verify_subset_distribution(self) -> Dict[int, int]:
        """Verify class distribution in subset is balanced."""
        if not self.config.use_subset:
            self.logger.info("Not using subset, skipping distribution verification")
            return {}
        
        self.logger.info("Verifying stratified subset distribution...")
        
        # Get targets for subset
        if hasattr(self.train_dataset.dataset, 'targets'):
            all_targets = np.array(self.train_dataset.dataset.targets)
        else:
            # ImageFolder doesn't have targets, extract from samples
            all_targets = np.array([s[1] for s in self.train_dataset.dataset.samples])
        
        subset_targets = all_targets[self.subset_indices]
        
        # Count samples per class
        unique, counts = np.unique(subset_targets, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # Log summary
        self.logger.info(f"Subset contains {len(unique)} classes")
        self.logger.info(f"Samples per class - Min: {min(counts)}, Max: {max(counts)}, "
                        f"Mean: {np.mean(counts):.1f}, Std: {np.std(counts):.1f}")
        
        return distribution

