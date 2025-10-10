#!/bin/bash
# Quick launcher for Linux/Mac - Gradio Version
# Run: chmod +x run_local.sh && ./run_local.sh

echo "============================================================"
echo "CIFAR-100 Classifier - Local Testing (Gradio)"
echo "============================================================"
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ ERROR: app.py not found!"
    echo "Please run this script from the CIFAR100HFS directory"
    exit 1
fi

# Check if model exists
if [ ! -f "cifar100_model.pth" ]; then
    echo "❌ ERROR: cifar100_model.pth not found!"
    echo "Please ensure the model file is in this directory"
    exit 1
fi

echo "✅ Starting Gradio app..."
echo ""
echo "🌐 Opening browser at: http://localhost:7860"
echo ""
echo "⚠️  Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

python app.py
