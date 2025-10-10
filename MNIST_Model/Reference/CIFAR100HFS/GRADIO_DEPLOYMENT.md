# 🎨 CIFAR-100 Gradio App - Deployment Guide

## ✅ Updated to Gradio!

Your app has been converted from Streamlit to **Gradio** for Hugging Face Spaces deployment.

---

## 🎯 Why Gradio?

✅ **Native HF Integration** - Built specifically for Hugging Face  
✅ **Simpler Deployment** - Less configuration needed  
✅ **Better Performance** - Faster loading times  
✅ **Easier Sharing** - One-click share links  
✅ **Beautiful UI** - Modern, clean interface out of the box  

---

## 🚀 Quick Start (3 Steps)

### Step 1: Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open browser at: `http://localhost:7860`

### Step 2: Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose:
   - **SDK**: Gradio
   - **Name**: cifar100-classifier
   - **Hardware**: CPU (free)

### Step 3: Upload Files

Upload these 6 files via web interface:
1. `app.py`
2. `model.py`
3. `cifar100_model.pth`
4. `requirements.txt`
5. `README.md`
6. `.gitattributes`

**Done!** Your app will be live in 2-3 minutes.

---

## 🎨 Gradio App Features

### User Interface
```
┌─────────────────────────────────────────────┐
│  🖼️ CIFAR-100 Image Classifier              │
├─────────────────────────────────────────────┤
│                                             │
│  📤 Upload Image      │  🎯 Results         │
│  ┌─────────────────┐  │  ┌────────────────┐ │
│  │                 │  │  │  DOLPHIN       │ │
│  │   [Your Image]  │  │  │  87.45%        │ │
│  │                 │  │  └────────────────┘ │
│  └─────────────────┘  │                     │
│  [Upload] [Webcam]    │  Top 10 Predictions:│
│  [Clipboard]          │  1. dolphin  87.45% │
│                       │  2. whale     5.23%  │
│  💡 Tips:             │  3. seal      3.12%  │
│  • Clear images       │  4. shark     1.87%  │
│  • Centered object    │  5. fish      0.95%  │
│                       │  ...                 │
└─────────────────────────────────────────────┘
```

### Key Features

✅ **Automatic Prediction** - Classifies on upload  
✅ **Top-10 Results** - Shows best predictions  
✅ **Confidence Bars** - Visual probability display  
✅ **Multiple Upload Options** - File, webcam, or clipboard  
✅ **Responsive Design** - Works on mobile  
✅ **Clean Interface** - No clutter, easy to use  

---

## 🔄 Changes from Streamlit

### What's Different

| Feature | Streamlit | Gradio |
|---------|-----------|--------|
| **Port** | 8501 | 7860 |
| **Run Command** | `streamlit run app.py` | `python app.py` |
| **SDK in README** | `sdk: streamlit` | `sdk: gradio` |
| **Dependencies** | streamlit | gradio |
| **Upload Options** | File only | File, webcam, clipboard |
| **Auto-predict** | Manual button | Automatic on upload |

### What Stayed the Same

✅ Model architecture (`model.py`)  
✅ Model weights (`cifar100_model.pth`)  
✅ Core functionality (classification)  
✅ 100 classes  
✅ Confidence scores  
✅ Top-K predictions  

---

## 📝 Updated Files

### Modified:
- ✅ `app.py` - Now uses Gradio instead of Streamlit
- ✅ `requirements.txt` - Changed streamlit to gradio
- ✅ `README.md` - Updated SDK to gradio
- ✅ `run_local.bat` - Updated port number
- ✅ `run_local.sh` - Updated port number
- ✅ `test_app_locally.py` - Updated for Gradio

### Unchanged:
- ✅ `model.py` - Same architecture
- ✅ `cifar100_model.pth` - Same model weights
- ✅ `.gitattributes` - Same LFS config

---

## 🧪 Testing

### Quick Test

```bash
python test_app_locally.py
```

### Manual Test

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python app.py

# 3. Open browser
# Go to: http://localhost:7860

# 4. Upload image

# 5. See results!
```

---

## ☁️ Deployment to Hugging Face

### Method 1: Web Interface (Easiest)

1. **Create Space**:
   - Go to https://huggingface.co/new-space
   - SDK: **Gradio**
   - Name: `cifar100-classifier`

2. **Upload Files**:
   - Click "Files" → "Upload files"
   - Upload all 6 core files
   - Commit changes

3. **Wait**: 2-3 minutes for build

4. **Access**: `https://huggingface.co/spaces/YOUR_USERNAME/cifar100-classifier`

### Method 2: Git CLI

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/cifar100-classifier
cd cifar100-classifier

# Copy files from CIFAR100HFS

git lfs install
git lfs track "*.pth"
git add .
git commit -m "Deploy CIFAR-100 Gradio app"
git push
```

---

## 🎨 Gradio-Specific Features

### Advantages

1. **Automatic Prediction**: No submit button needed
2. **Multiple Input Sources**: Upload, webcam, or paste
3. **Built-in Examples**: Easy to add example images
4. **Sharing**: One-click sharing with `share=True`
5. **API**: Automatic REST API generation
6. **Flagging**: Users can flag interesting results

### Using the API

Once deployed, your app automatically gets a REST API. To use it:

**Option 1: Install gradio_client (optional)**
```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/cifar100-classifier")
result = client.predict("path/to/image.jpg")
print(result)
```

**Option 2: Direct API calls**
Your deployed space will have API documentation at:
`https://huggingface.co/spaces/YOUR_USERNAME/cifar100-classifier?view=api`

---

## 🔧 Customization

### Change Theme

Edit `app.py` line 185:
```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Change to: gr.themes.Glass(), gr.themes.Monochrome(), etc.
```

### Add Examples

Create `examples/` folder with images, then edit `app.py` line 253:
```python
gr.Examples(
    examples=["examples/cat.jpg", "examples/car.jpg"],
    inputs=image_input,
)
```

### Change Port (Local Testing)

Edit `app.py` line 277:
```python
demo.launch(server_port=7860)  # Change to any port
```

### Enable Sharing

Edit `app.py` line 275:
```python
demo.launch(share=True)  # Creates temporary public link
```

---

## 📊 Performance

### Local Testing
- **Load Time**: 1-2 seconds
- **Inference**: 50-200ms (CPU), 10-30ms (GPU)
- **Memory**: ~500MB

### Hugging Face Spaces
- **First Load**: 3-5 seconds
- **Inference**: 100-500ms (CPU), 30-100ms (GPU)
- **Concurrent Users**: Supported (HF handles scaling)

---

## 🐛 Troubleshooting

### "Port already in use"
```bash
# Change port in app.py or kill existing process
```

### "Gradio not found"
```bash
pip install gradio>=4.0.0
```

### "Model predictions wrong"
- Verify model.py matches training architecture
- Check normalization values

### "Slow on HF Spaces"
- First load is always slower
- Consider GPU hardware upgrade
- Subsequent loads are faster

---

## 💡 Pro Tips

1. **Test Locally First**: Always verify before deploying
2. **Use Examples**: Add example images for users to try
3. **Monitor Logs**: Check HF Space logs for issues
4. **Update README**: Keep model accuracy and info current
5. **Version Control**: Use Git tags for different versions

---

## 🎯 Next Steps

### For Testing:
```bash
python test_app_locally.py
```

### For Deployment:
1. Create HF Space (Gradio SDK)
2. Upload 6 core files
3. Wait for build
4. Share your app!

---

## 📞 Support

- **Gradio Docs**: https://www.gradio.app/docs
- **HF Spaces**: https://huggingface.co/docs/hub/spaces
- **Examples**: https://huggingface.co/spaces (search "gradio")

---

**🎉 Your Gradio app is ready to deploy! Much simpler than Streamlit! 🚀**

*Last updated: October 10, 2025*

