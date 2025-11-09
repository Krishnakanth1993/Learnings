"""
Script to compute precision, recall, and F1 scores from validation dataset.

This script:
1. Loads the trained ResNet50 model
2. Loads ImageNet validation dataset
3. Runs inference on validation set
4. Computes precision, recall, F1 for each class
5. Saves metrics to metrics.json

Usage:
    python compute_metrics.py --model_path best_checkpoint_epoch094.pt --val_data_path /path/to/imagenet/val
"""

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys

# Import our modules
from model_core import load_model
from metrics import compute_metrics_from_validation
from imagenet_classes import IMAGENET_CLASSES


def get_imagenet_val_loader(val_data_path: str, batch_size: int = 32, num_workers: int = 4):
    """
    Create DataLoader for ImageNet validation dataset.
    
    Args:
        val_data_path: Path to ImageNet validation directory
        batch_size: Batch size for validation
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for validation dataset
    """
    # ImageNet normalization values
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Load ImageNet validation dataset
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_data_path,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return val_loader


def main():
    parser = argparse.ArgumentParser(description='Compute metrics from validation dataset')
    parser.add_argument('--model_path', type=str, default='best_checkpoint_epoch094.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='Path to ImageNet validation directory')
    parser.add_argument('--output_file', type=str, default='metrics.json',
                        help='Output file for metrics (default: metrics.json)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use (None = all, default: None)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file '{args.model_path}' not found.")
        sys.exit(1)
    
    # Check if validation data path exists
    if not os.path.exists(args.val_data_path):
        print(f"ERROR: Validation data path '{args.val_data_path}' not found.")
        sys.exit(1)
    
    print("="*60)
    print("Computing Metrics from Validation Dataset")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Validation Data: {args.val_data_path}")
    print(f"Output File: {args.output_file}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    try:
        model, device = load_model(args.model_path)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    try:
        val_loader = get_imagenet_val_loader(
            args.val_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Validation dataset loaded: {len(val_loader.dataset)} samples")
        
        # Limit number of samples if specified
        if args.num_samples is not None:
            # Create a subset of the dataset
            indices = list(range(min(args.num_samples, len(val_loader.dataset))))
            subset = torch.utils.data.Subset(val_loader.dataset, indices)
            val_loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            print(f"Using subset: {len(subset)} samples")
    except Exception as e:
        print(f"ERROR: Failed to load validation dataset: {e}")
        sys.exit(1)
    
    # Compute metrics
    print("\nComputing metrics...")
    try:
        metrics_manager = compute_metrics_from_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            class_names=IMAGENET_CLASSES,
            output_file=args.output_file,
            num_classes=1000
        )
        print("\nSUCCESS: Metrics computed and saved!")
        print(f"Metrics file: {args.output_file}")
    except Exception as e:
        print(f"ERROR: Failed to compute metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

