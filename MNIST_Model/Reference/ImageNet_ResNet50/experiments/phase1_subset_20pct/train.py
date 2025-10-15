"""
Phase 1: ImageNet ResNet-50 Training on 20% Subset
Train on stratified 20% subset to validate approach and find optimal hyperparameters.

Target: 50-60% top-1 accuracy on subset
Strategy: Fast iteration, LR finding, baseline performance

Author: Krishnakanth
Date: 2025-10-13
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

# Import services
from services.lr_finder import LRFinder

# Import utils
from utils.visualization import TrainingVisualizer
from utils.metrics import PerformanceTracker


def run_lr_finder(model, train_loader, device, save_dir):
    """Run LR finder before training."""
    print("\n" + "="*70)
    print("RUNNING LR FINDER")
    print("="*70)
    
    # Create temporary optimizer for LR finding
    temp_optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    criterion = F.nll_loss
    
    # Initialize LR finder
    lr_finder = LRFinder(model, temp_optimizer, criterion, device)
    
    # Run range test
    suggested_lr, suggested_max_lr = lr_finder.find_lr(
        train_loader,
        start_lr=1e-7,
        end_lr=1.0,
        num_iter=100,
        save_path=os.path.join(save_dir, 'lr_finder_results.png')
    )
    
    print("="*70)
    print(f"LR Finder completed!")
    print(f"  Suggested LR: {suggested_lr:.2e}")
    print(f"  Suggested Max LR: {suggested_max_lr:.2e}")
    print("="*70)
    
    return suggested_lr, suggested_max_lr


def main():
    """
    Main function for Phase 1 training.
    """
    print("="*70)
    print("IMAGENET RESNET-50 - PHASE 1: 20% SUBSET TRAINING")
    print("="*70)
    print("Goal: Validate approach and establish baseline performance")
    print("Target: 50-60% top-1 accuracy on 20% stratified subset")
    print("="*70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    config = Config()
    
    # Data configuration
    config.data.data_dir = '/home/ubuntu/data/imagenet'  # Adjust path as needed
    config.data.batch_size = 64  # Small batch for VRAM efficiency
    config.data.num_workers = 8
    #config.data.input_size = 128  # Start with smaller size for faster training
    config.data.use_subset = True
    config.data.subset_percentage = 0.2  # 20% stratified sampling
    config.data.subset_seed = 42
    config.data.cache_subset_indices = True
    
    # Model configuration
    config.model.model_name = 'resnet50'
    config.model.num_classes = 1000
    config.model.dropout_rate = 0.0  # No dropout initially
    config.model.use_pretrained = False  # Train from scratch
    
    # Training configuration
    config.training.epochs = 100  # Fast iteration on subset
    config.training.learning_rate = 0.05  # Will be updated by LR finder
    config.training.optimizer_type = 'SGD'
    config.training.weight_decay = 1e-4
    config.training.gradient_accumulation_steps = 4  # Effective batch = 32*4 = 128
    config.training.use_amp = True  # Mixed precision for speed
    config.training.max_grad_norm = 1.0
    config.training.label_smoothing = 0.1

    # Loss function configuration
    config.training.loss_function = 'CrossEntropyLoss'
    
    # Scheduler configuration
    config.training.scheduler_type = 'ReduceLROnPlateau'
    config.training.onecycle_max_lr = 0.003  # Will be updated by LR finder
    config.training.onecycle_pct_start = 0.3
    config.training.onecycle_div_factor = 5.0
    config.training.onecycle_final_div_factor = 1000.0
    
    # Checkpointing
    config.training.save_every_n_epochs = 5
    config.training.keep_best_n_checkpoints = 3
    config.training.early_stopping_patience = 10
    
    # Logging configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.logging.log_dir = os.path.join(script_dir, 'logs')
    config.logging.model_save_dir = os.path.join(script_dir, 'models')
    config.logging.experiment_name = 'phase1_subset_20pct'
    
    print("\nConfiguration Summary:")
    print(f"  Data Directory: {config.data.data_dir}")
    print(f"  Subset: {config.data.subset_percentage*100}% stratified")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.get_effective_batch_size()}")
    print(f"  Input Size: {config.data.input_size}x{config.data.input_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Mixed Precision: {config.training.use_amp}")
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
    
    logger.info("Setting up data...")
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
    
    logger.info("Building ResNet-50 model...")
    builder = ModelBuilder()
    model = builder.build_resnet50(config.model).build()
    model = model.to(device)
    
    # Get model summary
    model_info = model.get_model_summary((3, config.data.input_size, config.data.input_size), logger)
    
    # =========================================================================
    # LR FINDER (Optional - uncomment to run)
    # =========================================================================
    
    # Uncomment the following lines to run LR finder before training
    # suggested_lr, suggested_max_lr = run_lr_finder(model, train_loader, device, config.logging.log_dir)
    # 
    # # Update config with suggested values
    # if suggested_lr:
    #     config.training.learning_rate = suggested_lr
    #     config.training.onecycle_max_lr = suggested_max_lr / 10  # Conservative max LR
    #     logger.info(f"Updated LR from LR finder: {config.training.learning_rate:.2e}")
    #     logger.info(f"Updated max LR: {config.training.onecycle_max_lr:.2e}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    logger.info("Initializing trainer...")
    trainer = ImageNetTrainer(config.training, logger, config.logging.model_save_dir)
    
    logger.info("Starting training...")
    print("\n" + "="*70)
    print("TRAINING STARTED")
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
        print("PHASE 1 TRAINING COMPLETED!")
        print("="*70)
        print(f"Final Results:")
        print(f"  Train Top-1 Accuracy: {final_metrics['final_train_accuracy']:.2f}%")
        print(f"  Train Top-5 Accuracy: {final_metrics['final_train_top5_accuracy']:.2f}%")
        print(f"  Val Top-1 Accuracy: {final_metrics['final_test_accuracy']:.2f}%")
        print(f"  Val Top-5 Accuracy: {final_metrics['final_test_top5_accuracy']:.2f}%")
        print(f"  Best Val Top-1: {final_metrics['best_test_accuracy']:.2f}% (Epoch {final_metrics['best_epoch']})")
        print(f"  Best Val Top-5: {final_metrics['best_test_top5_accuracy']:.2f}%")
        print("="*70)
        print(f"\nNext Steps:")
        print(f"  1. Review training curves: {curves_path}")
        print(f"  2. Check best model: {config.logging.model_save_dir}/best_model.pth")
        print(f"  3. Run Grad-CAM analysis to identify improvements")
        print(f"  4. Proceed to Phase 2 with enhanced augmentation")
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

