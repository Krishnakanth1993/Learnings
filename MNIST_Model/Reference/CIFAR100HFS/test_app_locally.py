"""
Local Testing Script for CIFAR-100 Gradio App
Run this to test your app locally before deploying to Hugging Face

Usage:
    python test_app_locally.py
    
Then open: http://localhost:7860
"""

import subprocess
import sys
import os

def test_imports():
    """Test if all required packages are installed."""
    print("="*60)
    print("Testing imports...")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'gradio': 'Gradio',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'plotly': 'Plotly'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed!")
    return True


def test_model_file():
    """Test if model file exists."""
    print("\n" + "="*60)
    print("Testing model file...")
    print("="*60)
    
    model_path = "cifar100_model.pth"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model file NOT found: {model_path}")
        print("\nPlease ensure 'cifar100_model.pth' is in the current directory")
        return False


def test_model_loading():
    """Test if model loads correctly."""
    print("\n" + "="*60)
    print("Testing model loading...")
    print("="*60)
    
    try:
        import torch
        from model import CIFAR100ResNet34, ModelConfig
        
        config = ModelConfig(
            input_channels=3,
            input_size=(32, 32),
            num_classes=100,
            dropout_rate=0.05
        )
        
        model = CIFAR100ResNet34(config)
        
        # Try loading checkpoint
        checkpoint = torch.load("cifar100_model.pth", map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: torch.Size([1, 100])")
        
        if output.shape == torch.Size([1, 100]):
            print("✅ Output shape correct!")
            return True
        else:
            print("❌ Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_gradio():
    """Run Gradio app locally."""
    print("\n" + "="*60)
    print("Starting Gradio app...")
    print("="*60)
    print("\n🌐 Opening app in browser at: http://localhost:7860")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Gradio server stopped")


def main():
    """Main testing workflow."""
    print("\n" + "="*60)
    print("🧪 CIFAR-100 GRADIO APP - LOCAL TESTING")
    print("="*60)
    
    # Run tests
    imports_ok = test_imports()
    model_file_ok = test_model_file()
    model_load_ok = test_model_loading() if imports_ok and model_file_ok else False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"{'✅' if imports_ok else '❌'} Package imports")
    print(f"{'✅' if model_file_ok else '❌'} Model file exists")
    print(f"{'✅' if model_load_ok else '❌'} Model loads correctly")
    
    if imports_ok and model_file_ok and model_load_ok:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
        response = input("\n🚀 Run Gradio app locally? (y/n): ")
        if response.lower() == 'y':
            run_gradio()
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the app.")
        
        if not imports_ok:
            print("\n💡 To fix missing packages:")
            print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
