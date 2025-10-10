# 📚 Files Explained - What Each File Does

## 🎯 Essential Files (Must Upload to Hugging Face)

### 1. `app.py` (Main Application)
**What it does**: The Streamlit web application
**Key features**:
- Image upload interface
- Model loading and inference
- Results visualization
- Interactive charts
- Download functionality

**Don't modify unless**: You want to customize UI or add features

---

### 2. `model.py` (Model Architecture)
**What it does**: Defines the ResNet-34 neural network
**Contains**:
- `CIFAR100ResNet34` class
- `BasicBlock` and `BottleneckBlock` classes
- `ModelConfig` dataclass
- Weight initialization

**Don't modify**: This must match your trained model architecture

---

### 3. `cifar100_model.pth` (Trained Model Weights)
**What it does**: Your trained model parameters
**Size**: ~93 MB
**Contains**: Neural network weights from training

**Important**: Must be uploaded with Git LFS

---

### 4. `requirements.txt` (Dependencies)
**What it does**: Lists Python packages needed
**Contains**:
```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
Pillow>=9.5.0
numpy>=1.24.0
plotly>=5.17.0
```

**When to modify**: If you add new Python imports to `app.py`

---

### 5. `README.md` (Space Description)
**What it does**: Displays on your Hugging Face Space homepage
**Contains**:
- YAML metadata (title, emoji, SDK settings)
- Project description
- Features list
- Model details
- Usage instructions

**Customize**: Add your info, update accuracy numbers

---

### 6. `.gitattributes` (Git LFS Configuration)
**What it does**: Tells Git to use LFS for large files
**Contains**: Rules to track `*.pth` files

**Don't modify**: Unless you know what you're doing

---

## 🛠️ Development Files (Optional, Not Required for HF)

### 7. `test_app_locally.py` (Testing Script)
**What it does**: Automated testing before deployment
**Tests**:
- Package installation
- Model file presence
- Model loading
- Inference test

**Use when**: Before deploying to verify everything works

---

### 8. `run_local.bat` (Windows Launcher)
**What it does**: Quick launcher for Windows
**Usage**: Double-click to run app locally

---

### 9. `run_local.sh` (Linux/Mac Launcher)
**What it does**: Quick launcher for Unix systems
**Usage**: `chmod +x run_local.sh && ./run_local.sh`

---

### 10. `.gitignore` (Git Ignore Rules)
**What it does**: Tells Git which files to ignore
**Ignores**: Cache files, logs, test images, etc.

---

## 📖 Documentation Files

### 11. `DEPLOYMENT_GUIDE.md`
**For**: Detailed step-by-step deployment instructions
**Read when**: You're ready to deploy to Hugging Face

---

### 12. `QUICKSTART.md`
**For**: Quick reference for common tasks
**Read when**: You want a fast overview

---

### 13. `PROJECT_SUMMARY.md`
**For**: Complete project overview
**Contains**: All features, structure, customization options

---

### 14. `START_HERE.md`
**For**: First-time setup and orientation
**Read when**: You're starting fresh

---

### 15. `FILES_EXPLAINED.md` (This file!)
**For**: Understanding what each file does

---

## 🎯 Minimum Files for Hugging Face Deployment

Upload these 6 files:

1. ✅ `app.py`
2. ✅ `model.py`
3. ✅ `cifar100_model.pth`
4. ✅ `requirements.txt`
5. ✅ `README.md`
6. ✅ `.gitattributes`

Everything else is optional but helpful for development.

---

## 🔄 File Dependencies

```
app.py
  └── requires: model.py (imports CIFAR100ResNet34)
  └── requires: cifar100_model.pth (loads weights)
  └── requires: requirements.txt (dependencies)

model.py
  └── standalone (no dependencies on other project files)

cifar100_model.pth
  └── must match: model.py architecture

README.md
  └── used by: Hugging Face Space (homepage)

.gitattributes
  └── used by: Git LFS (for cifar100_model.pth)
```

---

## 📝 Editing Guide

### To Change App Appearance
→ Edit `app.py` (CSS section, lines 37-60)

### To Change Model
→ Replace `cifar100_model.pth` and update `model.py` if architecture changed

### To Add Dependencies
→ Edit `requirements.txt` and add package name

### To Update Documentation
→ Edit `README.md` (shows on HF Space homepage)

---

## 🎨 Quick Customization Examples

### Change Top-K Default
`app.py` line 97:
```python
value=5,  # Change to 3, 10, etc.
```

### Change Page Title
`app.py` line 29:
```python
page_title="Your Title Here",
```

### Change Emoji
`README.md` line 3:
```yaml
emoji: 🖼️  # Change to any emoji
```

### Change Color Theme
`README.md` lines 4-5:
```yaml
colorFrom: blue    # Options: red, green, yellow, pink, purple
colorTo: purple    # Options: indigo, teal, orange, lime
```

---

## 🔍 File Sizes

```
app.py              : ~16 KB
model.py            : ~7 KB
cifar100_model.pth  : ~93 MB  ← Large file (needs Git LFS)
requirements.txt    : ~100 bytes
README.md           : ~4 KB
.gitattributes      : ~200 bytes
```

**Total**: ~93 MB (mostly the model file)

---

## ⚡ Quick Actions

**Test locally**:
```bash
python test_app_locally.py
```

**Run app**:
```bash
streamlit run app.py
```

**Check model**:
```python
import torch
checkpoint = torch.load("cifar100_model.pth", map_location='cpu')
print(checkpoint.keys())
```

---

## 🎯 Next Steps

1. **Read**: `START_HERE.md` for overall guide
2. **Test**: Run `python test_app_locally.py`
3. **Deploy**: Follow `DEPLOYMENT_GUIDE.md`
4. **Enjoy**: Share your deployed app!

---

*Last updated: October 10, 2025*

