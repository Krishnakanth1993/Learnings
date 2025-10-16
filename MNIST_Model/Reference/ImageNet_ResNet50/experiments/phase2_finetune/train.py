"""
Phase 2: ImageNet ResNet-50 Fine-Tuning with Advanced Augmentation
Fine-tune Phase 1 model with dropout, Mixup, Cutmix, and RandAugment.

Improvements:
- Dropout: 0.02
- Mixup augmentation
- Cutmix augmentation
- RandAugment

Target: 65-70% top-1 accuracy with improved generalization
Strategy: Load Phase 1 best model and fine-tune with regularization

Author: Krishnakanth
Date: 2025-10-16
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from datetime import datetime

# Import core modules
from core.config import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
from core.logger import Logger
from core.data_manager import ImageNetDataManager
from core.model import ResNet50, ModelBuilder
from core.trainer import ImageNetTrainer

# Import utils
from utils.visualization import TrainingVisualizer
from utils.metrics import PerformanceTracker


def main():
    """
    Main function for Phase 2 fine-tuning with advanced augmentation.
    """
    print("="*70)
    print("IMAGENET RESNET-50 - PHASE 2: FINE-TUNING WITH ADVANCED AUGMENTATION")
    print("="*70)
    print("Goal: Fine-tune with Mixup, Cutmix, RandAugment, and Dropout")
    print("Target: 65-70% top-1 accuracy with improved generalization")
    print("="*70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    config = Config()
    
    # Data configuration
    config.data.data_dir = '../../data/imagenet'  # Adjust path as needed
    config.data.batch_size = 32  # Keep same as Phase 1
    config.data.num_workers = 8
    config.data.input_size = 224  # Increase to full ImageNet size
    config.data.use_subset = True  # Still using subset for Phase 2
    config.data.subset_percentage = 0.2  # Same 20% subset
    config.data.subset_seed = 42
    config.data.cache_subset_indices = True
    
    # Enable advanced augmentations
    config.data.use_randaugment = True
    config.data.randaugment_n = 2  # Number of augmentation transformations to apply
    config.data.randaugment_m = 9  # Magnitude of augmentations
    
    # Mixup and Cutmix (configured in training, not data)
    config.training.use_mixup = True
    config.training.mixup_alpha = 0.2
    config.training.use_cutmix = True
    config.training.cutmix_alpha = 1.0
    
    # Model configuration
    config.model.model_name = 'resnet50'
    config.model.num_classes = 1000
    config.model.dropout_rate = 0.02  # Add dropout for regularization
    config.model.use_pretrained = True  # Load Phase 1 checkpoint
    
    # Set path to Phase 1 best model
    phase1_model_path = '../phase1_subset_20pct/models/best_model.pth'
    if os.path.exists(phase1_model_path):
        config.model.pretrained_path = phase1_model_path
        print(f"\n✓ Found Phase 1 model: {phase1_model_path}")
    else:
        print(f"\n⚠ Warning: Phase 1 model not found at {phase1_model_path}")
        print("  Training from scratch instead")
        config.model.use_pretrained = False
    
    # Training configuration
    config.training.epochs = 50  # More epochs for fine-tuning
    config.training.learning_rate = 0.0001  # Lower LR for fine-tuning
    config.training.optimizer_type = 'AdamW'  # Better for fine-tuning
    config.training.weight_decay = 1e-4
    config.training.gradient_accumulation_steps = 4  # Effective batch = 32*4 = 128
    config.training.use_amp = True  # Mixed precision for speed
    config.training.max_grad_norm = 1.0
    config.training.label_smoothing = 0.1
    
    # Loss function configuration
    config.training.loss_function = 'CrossEntropyLoss'
    
    # Scheduler configuration - use Cosine for fine-tuning
    config.training.scheduler_type = 'CosineAnnealingLR'
    config.training.cosine_t_max = 50  # Match epochs
    config.training.cosine_eta_min = 1e-6
    
    # Checkpointing
    config.training.save_every_n_epochs = 5
    config.training.keep_best_n_checkpoints = 3
    config.training.early_stopping_patience = 15  # More patience for fine-tuning
    
    # Logging configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.logging.log_dir = os.path.join(script_dir, 'logs')
    config.logging.model_save_dir = os.path.join(script_dir, 'models')
    config.logging.experiment_name = 'phase2_finetune'
    
    print("\nConfiguration Summary:")
    print(f"  Data Directory: {config.data.data_dir}")
    print(f"  Subset: {config.data.subset_percentage*100}% stratified")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.get_effective_batch_size()}")
    print(f"  Input Size: {config.data.input_size}x{config.data.input_size}")
    print(f"  Dropout Rate: {config.model.dropout_rate}")
    print(f"  RandAugment: Enabled (N={config.data.randaugment_n}, M={config.data.randaugment_m})")
    print(f"  Mixup: {'Enabled' if config.training.use_mixup else 'Disabled'} (alpha={config.training.mixup_alpha})")
    print(f"  Cutmix: {'Enabled' if config.training.use_cutmix else 'Disabled'} (alpha={config.training.cutmix_alpha})")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Optimizer: {config.training.optimizer_type}")
    print(f"  Scheduler: {config.training.scheduler_type}")
    print("="*70)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    # Initialize logger
    logger = Logger(config.logging)
    logger.log_config_summary(config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    logger.info("Setting up data with advanced augmentation...")
    data_manager = ImageNetDataManager(config.data, logger)
    
    try:
        train_loader, val_loader = data_manager.load_data()
        data_stats = data_manager.get_data_statistics()
        
        # Verify stratified sampling
        if config.data.use_subset:
            distribution = data_manager.verify_subset_distribution()
    
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("\nPlease download ImageNet-1k dataset manually:")
        logger.error("1. Register at https://image-net.org/download.php")
        logger.error("2. Download ILSVRC2012 training and validation sets")
        logger.error("3. Extract to data/imagenet/train and data/imagenet/val")
        logger.error("4. Run this script again")
        return
    
    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    
    logger.info("Building ResNet-50 model with dropout...")
    builder = ModelBuilder()
    model = builder.build_resnet50(config.model).build()
    model = model.to(device)
    
    # Get model summary
    model_info = model.get_model_summary((3, config.data.input_size, config.data.input_size), logger)
    
    logger.info(f"Model dropout rate: {config.model.dropout_rate}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    logger.info("Initializing trainer with Mixup/Cutmix...")
    trainer = ImageNetTrainer(config.training, logger, config.logging.model_save_dir)
    
    logger.info("Starting Phase 2 fine-tuning...")
    print("\n" + "="*70)
    print("PHASE 2 FINE-TUNING STARTED")
    print("="*70)
    
    try:
        metrics = trainer.train(model, train_loader, val_loader)
        
        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        
        logger.info("Saving training results...")
        
        # Plot training curves
        curves_path = os.path.join(
            config.logging.log_dir,
            f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        TrainingVisualizer.plot_training_curves(
            metrics.train_losses,
            metrics.test_losses,
            metrics.train_accuracies,
            metrics.test_accuracies,
            metrics.train_top5_accuracies,
            metrics.test_top5_accuracies,
            metrics.learning_rates,
            save_path=curves_path
        )
        
        # Save metrics to CSV
        tracker = PerformanceTracker()
        for i in range(len(metrics.train_losses)):
            tracker.add_epoch(
                i + 1,
                metrics.train_losses[i],
                metrics.train_accuracies[i],
                metrics.train_top5_accuracies[i],
                metrics.test_losses[i],
                metrics.test_accuracies[i],
                metrics.test_top5_accuracies[i],
                metrics.learning_rates[i]
            )
        
        metrics_csv_path = os.path.join(config.logging.log_dir, 'training_metrics.csv')
        tracker.save_csv(metrics_csv_path)
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        
        final_metrics = metrics.get_final_metrics()
        
        print("\n" + "="*70)
        print("PHASE 2 FINE-TUNING COMPLETED!")
        print("="*70)
        print(f"Final Results:")
        print(f"  Train Top-1 Accuracy: {final_metrics['final_train_accuracy']:.2f}%")
        print(f"  Train Top-5 Accuracy: {final_metrics['final_train_top5_accuracy']:.2f}%")
        print(f"  Val Top-1 Accuracy: {final_metrics['final_test_accuracy']:.2f}%")
        print(f"  Val Top-5 Accuracy: {final_metrics['final_test_top5_accuracy']:.2f}%")
        print(f"  Best Val Top-1: {final_metrics['best_test_accuracy']:.2f}% (Epoch {final_metrics['best_epoch']})")
        print(f"  Best Val Top-5: {final_metrics['best_test_top5_accuracy']:.2f}%")
        print("="*70)
        print(f"\nImprovements Applied:")
        print(f"  ✓ Dropout (0.02)")
        print(f"  ✓ RandAugment (N={config.data.randaugment_n}, M={config.data.randaugment_m})")
        print(f"  ✓ Mixup (alpha={config.training.mixup_alpha})")
        print(f"  ✓ Cutmix (alpha={config.training.cutmix_alpha})")
        print(f"  ✓ Larger input size ({config.data.input_size}x{config.data.input_size})")
        print("="*70)
        print(f"\nNext Steps:")
        print(f"  1. Review training curves: {curves_path}")
        print(f"  2. Check best model: {config.logging.model_save_dir}/best_model.pth")
        print(f"  3. Compare with Phase 1 results")
        print(f"  4. Run Grad-CAM analysis on misclassifications")
        print(f"  5. Proceed to Phase 3 for full dataset training")
        print("="*70)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\nTraining interrupted. Checkpoints saved in models/ directory.")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

