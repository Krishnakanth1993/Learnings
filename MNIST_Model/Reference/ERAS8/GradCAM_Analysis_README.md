# Grad-CAM Analysis for CIFAR-100 Model

## Overview

This directory contains a comprehensive Grad-CAM (Gradient-weighted Class Activation Mapping) analysis of the deployed CIFAR-100 ResNet-34 model to understand and visualize misclassifications.

## Purpose

Grad-CAM helps answer:
- **WHERE** does the model look when making predictions?
- **WHY** does the model misclassify certain images?
- **WHICH** classes are most confused with each other?
- **HOW** can we improve the model based on visual evidence?

## Files

### Main Analysis Notebook
- **`GradCAM_Analysis.ipynb`**: Complete Grad-CAM analysis pipeline
  - Loads best model from `../CIFAR100HFS/cifar100_model.pth`
  - Runs predictions on entire CIFAR-100 test set
  - Generates Grad-CAM visualizations for worst predictions
  - Provides interactive exploration widget

### Output Directory: `gradcam_results/`

Generated files:
1. **`confusion_matrix_full.png`**: 100×100 confusion matrix heatmap
2. **`per_class_accuracy.csv`**: Detailed accuracy metrics per class
3. **`per_class_accuracy_interactive.html`**: Interactive bar chart (open in browser)
4. **`most_confused_pairs.csv`**: Top confused class pairs
5. **`top_confused_pairs.png`**: Visualization of top confusions
6. **`worst_predictions/`**: 100 PNG files (one per class) with Grad-CAM analysis
7. **`confused_pair_*.png`**: Detailed analysis of top 5 confused pairs
8. **`analysis_summary.txt`**: Comprehensive text summary
9. **`recommendations.txt`**: Actionable recommendations for improvement

## What is Grad-CAM?

Grad-CAM generates a heatmap showing which regions of an image the model focuses on when making a prediction.

**How it works:**
1. Forward pass through the model
2. Compute gradients of target class with respect to final conv layer
3. Weight the activation maps by gradients
4. Create heatmap showing important regions

**Interpretation:**
- **Red/Yellow (hot)**: High importance regions
- **Blue (cold)**: Low importance regions
- Helps understand if model focuses on correct features

## Notebook Structure

### Section 1: Setup (Cells 1-3)
- Import libraries
- Define constants and configuration
- Set up output directories

### Section 2: Grad-CAM Implementation (Cells 4-5)
- `GradCAMVisualizer` class
- Hooks into `model.layer4` (final conv layer)
- Methods for generating and overlaying heatmaps

### Section 3: Data Loading (Cells 6-7)
- Load trained model from checkpoint
- Load CIFAR-100 test dataset
- Set up data loaders

### Section 4: Prediction Collection (Cells 8-11)
- Run inference on all 10,000 test images
- Collect predictions with confidence scores
- Build confusion matrix
- Identify worst predictions per class

### Section 5: Overall Analysis (Cells 12-17)
- Confusion matrix visualization
- Per-class accuracy charts
- Top confused pairs analysis
- Interactive plotly charts

### Section 6: Per-Class Deep Dive (Cells 18-24)
- Generate Grad-CAM for all 100 classes
- Show best 3 and worst 3 predictions per class
- Save 100 PNG files with visualizations
- Display top 5 worst performing classes

### Section 7: Interactive Exploration (Cells 25-26)
- Dropdown widget to select any class
- Dynamic Grad-CAM visualization
- Statistics for selected class

### Section 8: Insights & Recommendations (Cells 27-35)
- Pattern analysis (within-category vs cross-category)
- High-confidence error analysis
- Detailed confused pair examples
- Actionable recommendations

## Key Features

### 1. Worst Prediction Analysis
For each class, identifies:
- **Misclassifications** (wrong predictions with high confidence)
- **Low confidence correct** (right but uncertain predictions)

Focus on WORST cases because they reveal:
- What features confuse the model
- Where model attention is misdirected
- Which similar classes are problematic

### 2. Dual Grad-CAM for Misclassifications
When image is misclassified, shows:
- **True class Grad-CAM**: Where model SHOULD look
- **Predicted class Grad-CAM**: Where model ACTUALLY looks
- **Difference**: Reveals attention mismatch

### 3. Interactive Exploration
- Browse all 100 classes
- See worst predictions on demand
- Understand model behavior per class

### 4. Comprehensive Statistics
- Per-class accuracy
- Confusion matrix
- Confidence distributions
- Category-level patterns

## How to Use

### Quick Start
```bash
# Navigate to ERAS8 directory
cd ERAS8

# Open notebook in Jupyter
jupyter notebook GradCAM_Analysis.ipynb

# Run all cells (Runtime → Run All)
# Or run cell by cell to explore interactively
```

### Exploring Results

1. **Overall confusion**: Check cells 12-17 for confusion matrix and top pairs
2. **Worst classes**: Cell 24 shows detailed analysis of 5 worst classes
3. **Any specific class**: Use interactive widget in cell 26
4. **Confused pairs**: Cells 31-32 show why specific classes are confused
5. **Saved files**: Browse `gradcam_results/` folder for offline analysis

### Interactive HTML Charts

Open in browser for interactive exploration:
```bash
# Windows
start gradcam_results/per_class_accuracy_interactive.html

# Mac/Linux
open gradcam_results/per_class_accuracy_interactive.html
```

## Example Insights

### What You'll Discover

**Class Confusion Patterns:**
- Animals confused with similar animals (leopard → cheetah)
- Trees confused with other trees (oak → maple)
- Vehicles confused within vehicle category

**Grad-CAM Revelations:**
- Model sometimes focuses on background instead of object
- Similar textures (spots, stripes) cause confusion
- Small objects challenging for 32×32 images

**Confidence Issues:**
- Some errors have >90% confidence (very wrong but very sure!)
- Low confidence on ambiguous images (model is uncertain)

## Expected Runtime

- **Full analysis**: ~10-15 minutes (with GPU)
  - Prediction collection: ~2 minutes
  - Grad-CAM generation (500 images): ~8-10 minutes
  - Visualization: ~2-3 minutes

- **Interactive exploration**: Instant (pre-computed)

## Requirements

```bash
pip install grad-cam opencv-python-headless plotly ipywidgets
```

Already included in notebook imports:
- pytorch-grad-cam
- matplotlib, seaborn
- plotly (interactive charts)
- ipywidgets (interactive dropdown)
- opencv (image processing)

## Troubleshooting

### Issue: "Module not found"
```bash
pip install grad-cam opencv-python-headless plotly ipywidgets
```

### Issue: "Model loading error"
- Ensure `CIFAR100HFS/cifar100_model.pth` exists
- Check MODEL_PATH in cell 3

### Issue: "CUDA out of memory"
- Reduce batch size in cell 7
- Use CPU: Change device to 'cpu'

### Issue: "Interactive widget not working"
- Enable widgets: `jupyter nbextension enable --py widgetsnbextension`
- Restart kernel

## Output Examples

### Per-Class Visualization
Each class gets a visualization showing:
```
Class: leopard (ID: 42) - Accuracy: 67.0%

BEST #1: [Original] [Grad-CAM Overlay] [Heatmap]
BEST #2: [Original] [Grad-CAM Overlay] [Heatmap]
BEST #3: [Original] [Grad-CAM Overlay] [Heatmap]

WORST #1: [Original] [True CAM] [Pred CAM]
         True: leopard → Predicted: cheetah (92%)
WORST #2: [Original] [True CAM] [Pred CAM]
         True: leopard → Predicted: tiger (85%)
WORST #3: [Original] [Grad-CAM] [Heatmap]
         Low confidence correct (23%)
```

### Confused Pair Analysis
```
Confusion: leopard → cheetah (12 times)

Example 1: [Original] [CAM(leopard)] [CAM(cheetah)] [Difference]
Example 2: [Original] [CAM(leopard)] [CAM(cheetah)] [Difference]
Example 3: [Original] [CAM(leopard)] [CAM(cheetah)] [Difference]

Shows how model focuses on spots pattern for both, causing confusion.
```

## Next Steps

After running analysis:

1. **Review worst performing classes** (cell 24 output)
2. **Check confused pairs** to understand common mistakes
3. **Use interactive widget** to deep-dive into specific classes
4. **Read recommendations.txt** for improvement ideas
5. **Implement suggested changes** in next training iteration

## Technical Notes

### Why layer4 for Grad-CAM?
- `layer4` is the final conv layer before global pooling
- Contains highest-level semantic features
- Best balance of spatial resolution and semantic meaning
- Standard choice for ResNet architectures

### Why focus on worst predictions?
- Best predictions don't reveal much (model already correct)
- Worst predictions show model's weaknesses
- Understanding failures drives improvements
- Targeted analysis more actionable than random sampling

### Confidence score interpretation
- **>90%**: Very confident (check if justified)
- **70-90%**: Confident (typical for correct predictions)
- **50-70%**: Moderate (model uncertain)
- **<50%**: Low confidence (model confused)

## References

- **Grad-CAM Paper**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **pytorch-grad-cam**: https://github.com/jacobgil/pytorch-grad-cam

---

**Created**: October 11, 2025
**Author**: Krishnakanth
**Model**: CIFAR-100 ResNet-34
**Purpose**: Understand and improve model performance through visual analysis

