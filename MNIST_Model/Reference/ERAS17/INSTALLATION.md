# Installation Guide

Complete setup instructions for the Autonomous Car Navigation project.

---

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **GPU**: Optional (CPU-only training works fine for this project)

### Check Python Version

```bash
python --version
# or
python3 --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

---

## 🚀 Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Clone or navigate to the project directory
cd ERAS17

# Install required packages
pip install torch torchvision
pip install PyQt6
pip install numpy
```

### Option 2: Using requirements.txt

Create a `requirements.txt` file:

```txt
torch>=2.0.0
torchvision>=0.15.0
PyQt6>=6.4.0
numpy>=1.23.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Option 3: Using conda

```bash
# Create a new conda environment
conda create -n car_nav python=3.10

# Activate the environment
conda activate car_nav

# Install PyTorch (CPU version)
conda install pytorch torchvision cpuonly -c pytorch

# Install PyQt6 and numpy
pip install PyQt6 numpy
```

---

## 🔧 Detailed Installation Steps

### Step 1: Set Up Virtual Environment (Recommended)

#### On Windows:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate
```

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### Step 2: Install PyTorch

#### CPU-Only Installation:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### GPU Installation (NVIDIA CUDA):

Check your CUDA version:

```bash
nvidia-smi
```

Then install the appropriate PyTorch version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for more options.

### Step 3: Install PyQt6

```bash
pip install PyQt6
```

### Step 4: Install NumPy

```bash
pip install numpy
```

### Step 5: Verify Installation

Create a test script `test_install.py`:

```python
import sys
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PyQt6 installed: {QApplication is not None}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Run the test:

```bash
python test_install.py
```

Expected output:

```
Python version: 3.10.x
PyTorch version: 2.x.x
NumPy version: 1.x.x
PyQt6 installed: True
CUDA available: False  # or True if GPU is available
```

---

## 📦 Package Versions

### Tested Configurations

| Package | Minimum Version | Recommended Version |
|---------|----------------|---------------------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0.0 | 2.1.0+ |
| PyQt6 | 6.4.0 | 6.6.0+ |
| NumPy | 1.23.0 | 1.24.0+ |

---

## 🐛 Troubleshooting Installation

### Issue 1: PyQt6 Import Error

**Error:**
```
ImportError: cannot import name 'QApplication' from 'PyQt6.QtWidgets'
```

**Solution:**
```bash
pip uninstall PyQt6
pip install PyQt6 --no-cache-dir
```

### Issue 2: PyTorch CUDA Mismatch

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
Reinstall PyTorch with the correct CUDA version:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: NumPy Version Conflict

**Error:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
pip install --upgrade numpy
```

### Issue 4: Permission Denied (Linux/macOS)

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
Use `--user` flag:

```bash
pip install --user torch torchvision PyQt6 numpy
```

### Issue 5: SSL Certificate Error

**Error:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch torchvision PyQt6 numpy
```

---

## 🔍 Verifying the Installation

### Run the Application

```bash
python Autonomous_DrivingRL.py
```

You should see:
1. A window with a map canvas
2. Control panel on the right
3. Reward chart at the bottom

### Test Basic Functionality

1. **Click on the map** to place the car
2. **Click again** to place a target
3. **Right-click** to finish placing targets
4. **Press SPACE** or click "START" to begin training

If the car starts moving and the reward chart updates, the installation is successful! ✅

---

## 🌐 Alternative Installation Methods

### Using UV (Modern Python Package Manager)

If you have `uv` installed:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install torch torchvision PyQt6 numpy
```

### Using Poetry

Create `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.8"
torch = "^2.0.0"
torchvision = "^0.15.0"
PyQt6 = "^6.4.0"
numpy = "^1.23.0"
```

Install:

```bash
poetry install
poetry run python Autonomous_DrivingRL.py
```

---

## 📝 Post-Installation Setup

### Download a Custom Map (Optional)

Place your custom map image in the project directory:

```bash
# Supported formats: PNG, JPG
# Recommended size: 600x600 to 1200x1200 pixels
# Black pixels = walls, White pixels = drivable area
```

### Configure Hyperparameters (Optional)

Edit `Autonomous_DrivingRL.py` to adjust training parameters:

```python
# Line ~30-45
BATCH_SIZE = 64      # Increase for more stable training
LR = 0.0005          # Decrease if training is unstable
GAMMA = 0.99         # Discount factor
```

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check Python version**: Ensure Python 3.8+
2. **Update pip**: `pip install --upgrade pip`
3. **Clear cache**: `pip cache purge`
4. **Reinstall from scratch**: Delete `.venv` and start over

For persistent issues, check:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (verify with `import torch`)
- [ ] PyQt6 installed (verify with `from PyQt6.QtWidgets import QApplication`)
- [ ] NumPy installed (verify with `import numpy`)
- [ ] `Autonomous_DrivingRL.py` runs without errors
- [ ] GUI window appears correctly
- [ ] Can place car and targets on the map

---

**Next Steps**: See [USAGE.md](USAGE.md) for how to use the application.

**Last Updated**: January 15, 2026
