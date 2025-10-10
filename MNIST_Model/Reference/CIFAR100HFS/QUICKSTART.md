# 🚀 Quick Start Guide

## Test Locally (Recommended First)

### Option 1: Using the test script

```bash
python test_app_locally.py
```

This will:
1. ✅ Check all dependencies
2. ✅ Verify model file exists
3. ✅ Test model loading
4. ✅ Launch Gradio app

### Option 2: Direct Gradio run

```bash
python app.py
```

Then open your browser to: `http://localhost:7860`

## Deploy to Hugging Face Spaces

### Quick Deployment (3 steps)

1. **Create Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - Choose Gradio SDK
   - Name it (e.g., `cifar100-classifier`)

2. **Clone and add files**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   
   # Copy all files from CIFAR100HFS folder
   # Then:
   git lfs install
   git lfs track "*.pth"
   git add .
   git commit -m "Deploy CIFAR-100 classifier"
   git push
   ```

3. **Done!** Your app will be live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## 📁 Files Overview

```
CIFAR100HFS/
├── app.py                    # Main Streamlit application
├── model.py                  # ResNet-34 architecture
├── cifar100_model.pth        # Your trained model weights
├── requirements.txt          # Python dependencies
├── README.md                 # Space description (for HF)
├── .gitattributes           # Git LFS configuration
├── test_app_locally.py      # Local testing script
├── DEPLOYMENT_GUIDE.md      # Detailed deployment steps
└── QUICKSTART.md            # This file
```

## 🎯 What Your App Does

- 📤 Upload images (JPG, PNG, BMP, WEBP)
- 🤖 AI classifies into 100 categories
- 📊 Shows top predictions with probabilities
- 📈 Interactive probability charts
- 💾 Download results as text

## 🐛 Common Issues

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"Model file not found"**
- Ensure `cifar100_model.pth` is in the same folder as `app.py`

**"CUDA out of memory"**
- The app automatically falls back to CPU if GPU memory is insufficient

## 💡 Tips

- Test locally before deploying
- Use Git LFS for model files >10MB
- Check Hugging Face Space logs if issues occur
- CPU inference works fine for single images

---

**Ready to deploy? Follow the steps above! 🚀**

