"""
Export clean model weights (weights only, no training config)
Fixed for PyTorch 2.6+ with custom class handling
"""
import torch
from model import CIFAR100ResNet34, ModelConfig
import os

# Define dummy TrainingConfig class to satisfy unpickler
# This matches what was saved during training
class TrainingConfig:
    """Dummy class to allow unpickling the checkpoint."""
    pass

# Register the safe global for PyTorch 2.6+
torch.serialization.add_safe_globals([TrainingConfig])

# Now load the checkpoint
print("Loading checkpoint...")
try:
    checkpoint = torch.load("cifar100_model.pth", map_location='cpu', weights_only=False)
    print(f"✅ Checkpoint loaded")
    print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("\nTrying alternative method...")
    # Alternative: Just load the state dict directly
    checkpoint = {'model_state_dict': torch.load("cifar100_model.pth", map_location='cpu', weights_only=False)}

# Create model
print("\nCreating model...")
config = ModelConfig(
    input_channels=3,
    input_size=(32, 32),
    num_classes=100,
    dropout_rate=0.05
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