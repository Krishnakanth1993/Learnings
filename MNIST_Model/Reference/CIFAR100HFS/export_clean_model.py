"""
Export clean model weights (weights only, no training config)
Fixed for PyTorch 2.6+ with custom class handling
"""
import torch
from model import CIFAR100ResNet34, ModelConfig
import os
import sys
import types
from dataclasses import dataclass

# Define dummy TrainingConfig class to satisfy unpickler
# This matches what was saved during training
@dataclass
class TrainingConfig:
    """Dummy class to allow unpickling the checkpoint."""
    epochs: int = 100
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    seed: int = 1
    optimizer_type: str = 'Adam'
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-08
    rmsprop_alpha: float = 0.99
    scheduler_type: str = 'OneCycleLR'
    cosine_t_max: int = 20
    exponential_gamma: float = 0.95
    plateau_mode: str = 'min'
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 0.0001
    onecycle_max_lr: float = 0.003
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 5
    onecycle_final_div_factor: float = 1000.0
    onecycle_anneal_strategy: str = 'cos'

# Create and register dummy module for unpickling
cifar100_training_module = types.ModuleType('cifar100_training')
cifar100_training_module.TrainingConfig = TrainingConfig
sys.modules['cifar100_training'] = cifar100_training_module

# Register the safe global for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([TrainingConfig])
except:
    pass  # Older PyTorch versions don't have this

# Now load the checkpoint
print("Loading checkpoint...")
try:
    checkpoint = torch.load("cifar100_model_20251011_093931.pth", map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint loaded")
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create model
print("\nCreating model...")
config = ModelConfig(
    input_channels=3,
    input_size=(32, 32),
    num_classes=100,
    dropout_rate=0.0  # Model was trained with no dropout
)

model = CIFAR100ResNet34(config)

# Load weights
print("Loading model weights...")
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded from model_state_dict")
else:
    model.load_state_dict(checkpoint)
    print("✅ Loaded directly")

# Test model works
print("\nTesting model...")
test_input = torch.randn(1, 3, 32, 32)
model.eval()
with torch.no_grad():
    output = model(test_input)
print(f"✅ Model test successful, output shape: {output.shape}")

# Save ONLY the model weights (clean, no extra classes)
print("\nSaving clean model...")
torch.save({
    'model_state_dict': model.state_dict()
}, "cifar100_model_clean.pth")

print("\n" + "="*60)
print("✅ CLEAN MODEL EXPORTED SUCCESSFULLY!")
print("="*60)
print(f"Output file: cifar100_model_clean.pth")
print(f"Size: {os.path.getsize('cifar100_model_clean.pth') / (1024*1024):.2f} MB")
print(f"\nNow update app.py to use: cifar100_model_clean.pth")
print("="*60)