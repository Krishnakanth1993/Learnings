"""
Compute Confusion Matrix for CIFAR-100 Model
Generates confusion matrix and various visualizations

Author: Krishnakanth
Date: 2025-10-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import model
sys.path.append('.')

# Import your model - adjust path based on which model you want to evaluate
from Model_Evolution.FineTune.model import CIFAR100ResNet34 as CIFAR100Model, ModelConfig

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# CIFAR-100 normalization
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_model_and_data(model_path, batch_size=128):
    """Load model and test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    config = ModelConfig(
        input_channels=3,
        input_size=(32, 32),
        num_classes=100,
        dropout_rate=0.05
    )
    
    model = CIFAR100Model(config)
    
    # Load weights
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'metrics' in checkpoint:
            print(f"Model metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    # Load CIFAR-100 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])
    
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Test data loaded: {len(test_dataset)} samples")
    
    return model, test_loader, device


def get_predictions(model, test_loader, device):
    """Get all predictions and true labels."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nGetting predictions on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing batches"):
            images = images.to(device)
            outputs = model(images)
            
            # Get predictions
            probs = torch.exp(outputs)  # Convert log_softmax to probabilities
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"✅ Predictions complete: {len(all_preds)} samples")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# Remove this import:
# import seaborn as sns

# Replace plot_confusion_matrix function with:
def plot_confusion_matrix(cm, save_path='confusion_matrix_full.png'):
    """Plot full 100x100 confusion matrix using matplotlib only."""
    print("\nPlotting full confusion matrix...")
    
    plt.figure(figsize=(24, 22))
    
    # Use imshow instead of seaborn heatmap
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Samples', fontsize=12, fontweight='bold')
    
    # Set ticks
    plt.xticks(range(100), CIFAR100_CLASSES, rotation=90, fontsize=7)
    plt.yticks(range(100), CIFAR100_CLASSES, rotation=0, fontsize=7)
    
    plt.title('CIFAR-100 Confusion Matrix (100×100)\nDiagonal = Correct Predictions', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_confusion_matrix_normalized(cm, save_path='confusion_matrix_normalized.png'):
    """Plot normalized confusion matrix using matplotlib only."""
    print("Plotting normalized confusion matrix...")
    
    # Normalize by row
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(24, 22))
    
    # Use imshow with RdYlGn colormap
    im = plt.imshow(cm_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Accuracy (per class)', fontsize=12, fontweight='bold')
    
    # Set ticks
    plt.xticks(range(100), CIFAR100_CLASSES, rotation=90, fontsize=7)
    plt.yticks(range(100), CIFAR100_CLASSES, rotation=0, fontsize=7)
    
    plt.title('CIFAR-100 Normalized Confusion Matrix\n(Row-wise: True Class Distribution)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_per_class_accuracy(cm, save_path='per_class_accuracy.png'):
    """Plot per-class accuracy bar chart."""
    print("Plotting per-class accuracy...")
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracy)
    sorted_classes = [CIFAR100_CLASSES[i] for i in sorted_indices]
    sorted_accuracy = class_accuracy[sorted_indices] * 100
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10))
    
    # Plot 1: All classes
    colors = ['#28a745' if acc > 80 else '#ffc107' if acc > 60 else '#dc3545' 
              for acc in sorted_accuracy]
    
    ax1.barh(range(len(sorted_classes)), sorted_accuracy, color=colors)
    ax1.set_yticks(range(len(sorted_classes)))
    ax1.set_yticklabels(sorted_classes, fontsize=6)
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Accuracy\n(All 100 Classes, Sorted)', fontsize=14, fontweight='bold')
    ax1.axvline(x=class_accuracy.mean()*100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {class_accuracy.mean()*100:.2f}%')
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Top 15 best
    ax2.barh(range(15), sorted_accuracy[-15:], color='#28a745')
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(sorted_classes[-15:], fontsize=10)
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Top 15 Best Performing Classes', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, acc in enumerate(sorted_accuracy[-15:]):
        ax2.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)
    
    # Plot 3: Top 15 worst
    ax3.barh(range(15), sorted_accuracy[:15], color='#dc3545')
    ax3.set_yticks(range(15))
    ax3.set_yticklabels(sorted_classes[:15], fontsize=10)
    ax3.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Top 15 Worst Performing Classes', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    for i, acc in enumerate(sorted_accuracy[:15]):
        ax3.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    return class_accuracy


def find_most_confused_pairs(cm, top_n=15):
    """Find the most confused class pairs."""
    confused_pairs = []
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j:  # Skip diagonal (correct predictions)
                confused_pairs.append({
                    'true_class': CIFAR100_CLASSES[i],
                    'predicted_class': CIFAR100_CLASSES[j],
                    'count': cm[i, j],
                    'true_idx': i,
                    'pred_idx': j
                })
    
    # Sort by count
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n{'='*90}")
    print(f"TOP {top_n} MOST CONFUSED CLASS PAIRS")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'True Class':<20} {'Predicted As':<20} {'Count':<10} {'% of True'}")
    print(f"{'-'*90}")
    
    for idx, pair in enumerate(confused_pairs[:top_n], 1):
        total_true = cm[pair['true_idx']].sum()
        percentage = (pair['count'] / total_true) * 100 if total_true > 0 else 0
        print(f"{idx:<6} {pair['true_class']:<20} {pair['predicted_class']:<20} "
              f"{pair['count']:<10} {percentage:.2f}%")
    
    print(f"{'='*90}\n")
    
    return confused_pairs[:top_n]


def generate_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    """Generate detailed classification report."""
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CIFAR100_CLASSES,
        digits=4,
        zero_division=0
    )
    
    print(f"\n{'='*90}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*90}")
    print(report)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write("CIFAR-100 Classification Report\n")
        f.write("="*90 + "\n\n")
        f.write(report)
    
    print(f"\n✅ Classification report saved: {save_path}")
    
    return report


def compute_top_k_accuracy(y_true, y_probs, k=5):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    top_k_acc = (correct / len(y_true)) * 100
    return top_k_acc


def analyze_model_performance(y_true, y_pred, y_probs):
    """Comprehensive performance analysis."""
    print(f"\n{'='*90}")
    print("MODEL PERFORMANCE ANALYSIS")
    print(f"{'='*90}")
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy:     {accuracy:.2f}%")
    print(f"  Top-1 Error:        {100-accuracy:.2f}%")
    print(f"  Top-5 Accuracy:     {compute_top_k_accuracy(y_true, y_probs, k=5):.2f}%")
    print(f"  Top-10 Accuracy:    {compute_top_k_accuracy(y_true, y_probs, k=10):.2f}%")
    print(f"\n  Macro Precision:    {precision*100:.2f}%")
    print(f"  Macro Recall:       {recall*100:.2f}%")
    print(f"  Macro F1-Score:     {f1*100:.2f}%")
    
    # Per-class statistics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Find best and worst classes
    best_idx = np.argmax(f1_per_class)
    worst_idx = np.argmin(f1_per_class)
    
    print(f"\nBest Performing Class:")
    print(f"  {CIFAR100_CLASSES[best_idx]:20s}: F1={f1_per_class[best_idx]*100:.2f}%, "
          f"Precision={precision_per_class[best_idx]*100:.2f}%, "
          f"Recall={recall_per_class[best_idx]*100:.2f}%")
    
    print(f"\nWorst Performing Class:")
    print(f"  {CIFAR100_CLASSES[worst_idx]:20s}: F1={f1_per_class[worst_idx]*100:.2f}%, "
          f"Precision={precision_per_class[worst_idx]*100:.2f}%, "
          f"Recall={recall_per_class[worst_idx]*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'top1_error': 100 - accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'top5_accuracy': compute_top_k_accuracy(y_true, y_probs, k=5),
        'top10_accuracy': compute_top_k_accuracy(y_true, y_probs, k=10)
    }


def save_per_class_metrics(y_true, y_pred, save_path='per_class_metrics.csv'):
    """Save per-class metrics to CSV."""
    print("\nComputing per-class metrics...")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Compute per-class accuracy
    class_accuracy = []
    for i in range(100):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum() * 100
        else:
            acc = 0
        class_accuracy.append(acc)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': CIFAR100_CLASSES,
        'Accuracy (%)': class_accuracy,
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100,
        'F1-Score (%)': f1 * 100,
        'Support': support
    })
    
    # Sort by F1-Score
    df = df.sort_values('F1-Score (%)', ascending=False)
    
    # Save
    df.to_csv(save_path, index=False)
    print(f"✅ Saved: {save_path}")
    
    # Display top 10 and bottom 10
    print(f"\n{'='*90}")
    print("TOP 10 BEST PERFORMING CLASSES")
    print(f"{'='*90}")
    print(df.head(10).to_string(index=False))
    
    print(f"\n{'='*90}")
    print("TOP 10 WORST PERFORMING CLASSES")
    print(f"{'='*90}")
    print(df.tail(10).to_string(index=False))
    
    return df


def main():
    """Main function to compute confusion matrix and metrics."""
    print("="*90)
    print("  CIFAR-100 CONFUSION MATRIX & PERFORMANCE ANALYSIS")
    print("="*90)
    
    # Get model path from user or use default
    import argparse
    parser = argparse.ArgumentParser(description='Compute confusion matrix for CIFAR-100 model')
    parser.add_argument('--model', type=str, 
                       default='Model_Evolution/FineTune/models/cifar100_model_20251007_083729.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default='./confusion_matrix_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, test_loader, device = load_model_and_data(args.model, args.batch_size)
    
    # Get predictions
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    print(f"✅ Confusion matrix computed: {cm.shape}")
    
    # Save confusion matrix as numpy array
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), cm)
    print(f"✅ Confusion matrix array saved: confusion_matrix.npy")
    
    # Overall performance
    metrics = analyze_model_performance(y_true, y_pred, y_probs)
    
    # Generate visualizations
    print(f"\n{'='*90}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*90}")
    
    plot_confusion_matrix(cm, 
                         save_path=os.path.join(args.output_dir, 'confusion_matrix_full.png'))
    
    plot_confusion_matrix_normalized(cm, 
                                    save_path=os.path.join(args.output_dir, 'confusion_matrix_normalized.png'))
    
    plot_per_class_accuracy(cm, 
                           save_path=os.path.join(args.output_dir, 'per_class_accuracy.png'))
    
    # Analysis
    print(f"\n{'='*90}")
    print("ANALYZING CONFUSION PATTERNS")
    print(f"{'='*90}")
    
    confused_pairs = find_most_confused_pairs(cm, top_n=20)
    
    # Generate reports
    generate_classification_report(y_true, y_pred, 
                                  save_path=os.path.join(args.output_dir, 'classification_report.txt'))
    
    save_per_class_metrics(y_true, y_pred, 
                          save_path=os.path.join(args.output_dir, 'per_class_metrics.csv'))
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CIFAR-100 Model Performance Summary\n")
        f.write("="*90 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write(f"Total Classes: 100\n\n")
        f.write(f"Top-1 Accuracy:  {metrics['accuracy']:.2f}%\n")
        f.write(f"Top-1 Error:     {metrics['top1_error']:.2f}%\n")
        f.write(f"Top-5 Accuracy:  {metrics['top5_accuracy']:.2f}%\n")
        f.write(f"Top-10 Accuracy: {metrics['top10_accuracy']:.2f}%\n\n")
        f.write(f"Macro Precision: {metrics['precision']:.2f}%\n")
        f.write(f"Macro Recall:    {metrics['recall']:.2f}%\n")
        f.write(f"Macro F1-Score:  {metrics['f1']:.2f}%\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Final summary
    print(f"\n{'='*90}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*90}")
    print(f"\nResults saved in: {args.output_dir}/")
    print(f"\nFiles generated:")
    print(f"  📊 confusion_matrix_full.png")
    print(f"  📊 confusion_matrix_normalized.png")
    print(f"  📊 per_class_accuracy.png")
    print(f"  📄 confusion_matrix.npy")
    print(f"  📄 classification_report.txt")
    print(f"  📄 per_class_metrics.csv")
    print(f"  📄 summary.txt")
    print(f"\n{'='*90}")
    print(f"Top-1 Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Top-1 Error:    {metrics['top1_error']:.2f}%")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()