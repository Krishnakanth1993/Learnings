#!/bin/bash

# EC2 Instance Setup Script for ImageNet ResNet50 Project
# Usage: bash ec2_setup.sh

set -e  # Exit on any error

echo "=========================================="
echo "EC2 Instance Setup - PyTorch Environment"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$HOME/S9"
REPO_URL="https://github.com/Krishnakanth1993/Learnings.git"
PYTORCH_ENV="/opt/pytorch/bin/activate"
KERNEL_NAME="pytorch_2_8"
KERNEL_DISPLAY="PyTorch_env"

# Step 1: Update system packages
echo -e "${BLUE}[1/8] Updating system packages...${NC}"
sudo apt-get update -y
sudo apt-get install -y unzip wget git

# Step 2: Activate PyTorch virtual environment
echo -e "${BLUE}[2/8] Activating PyTorch environment...${NC}"
source "$PYTORCH_ENV"
echo "PyTorch environment activated"

# Step 3: Upgrade pip
echo -e "${BLUE}[3/8] Upgrading pip...${NC}"
pip install --upgrade pip

# Step 4: Install and register Jupyter kernel
echo -e "${BLUE}[4/8] Setting up Jupyter kernel...${NC}"
pip install ipykernel
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$KERNEL_DISPLAY"
echo "Jupyter kernel '$KERNEL_DISPLAY' registered"

# Step 5: Create project directory and clone repository
echo -e "${BLUE}[5/8] Setting up project directory...${NC}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [ -d "Learnings" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd Learnings
    git pull
    cd "$PROJECT_DIR"
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
fi

# Step 6: Navigate to project directory
cd "$PROJECT_DIR/Learnings/MNIST_Model/Reference/ImageNet_ResNet50/"
echo "Current directory: $(pwd)"

# Step 7: Install Python requirements
echo -e "${BLUE}[6/8] Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Step 8: Install additional packages
echo -e "${BLUE}[7/8] Installing additional packages...${NC}"
pip install --index-url https://pypi.org/simple/ grad-cam

# Step 9: Create data directories
echo -e "${BLUE}[8/8] Creating data directories...${NC}"
mkdir -p data/imagenet
mkdir -p experiments
mkdir -p logs

# Verify installations
echo -e "\n${GREEN}=========================================="
echo "Setup Complete! Verification:"
echo "==========================================${NC}"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not found')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""
pip list | grep -E "torch|grad-cam|jupyter" || true
echo ""
echo "Jupyter kernels:"
jupyter kernelspec list
echo ""
echo -e "${GREEN}Project directory: $PROJECT_DIR/Learnings/MNIST_Model/Reference/ImageNet_ResNet50/${NC}"
echo -e "${GREEN}Data directory: $(pwd)/data/imagenet${NC}"
echo ""
echo "To activate the environment manually:"
echo "  source $PYTORCH_ENV"
echo ""
echo "Setup completed successfully!"