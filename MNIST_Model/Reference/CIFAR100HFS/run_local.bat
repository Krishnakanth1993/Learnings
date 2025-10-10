@echo off
REM Quick launcher for Windows - Gradio Version
REM Double-click this file to run the app locally

echo ============================================================
echo CIFAR-100 Classifier - Local Testing (Gradio)
echo ============================================================
echo.

REM Check if in correct directory
if not exist app.py (
    echo ERROR: app.py not found!
    echo Please run this script from the CIFAR100HFS directory
    pause
    exit /b 1
)

REM Check if model exists
if not exist cifar100_model.pth (
    echo ERROR: cifar100_model.pth not found!
    echo Please ensure the model file is in this directory
    pause
    exit /b 1
)

echo Starting Gradio app...
echo.
echo Opening browser at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python app.py

pause
