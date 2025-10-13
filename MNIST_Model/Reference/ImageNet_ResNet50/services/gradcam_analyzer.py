"""
Grad-CAM Analysis Service for ImageNet Models
Provides class activation mapping for model interpretability.

Author: Krishnakanth
Date: 2025-10-13
"""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAMAnalyzer:
    """
    Grad-CAM analyzer for ResNet-50 models.
    Provides visual explanations of model predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 device: torch.device, class_names: List[str]):
        """
        Initialize Grad-CAM analyzer.
        
        Args:
            model: Trained model
            target_layer: Layer to compute gradients from (e.g., model.layer4)
            device: Device model is on
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        
        # Initialize Grad-CAM
        self.cam = GradCAM(model=model, target_layers=[target_layer])
        
        # ImageNet normalization params for denormalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor to [0, 1] range.
        
        Args:
            tensor: Normalized image tensor (C, H, W)
        
        Returns:
            Denormalized numpy array (H, W, C)
        """
        # Convert to numpy and transpose
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        img = img * self.std + self.mean
        img = np.clip(img, 0, 1)
        
        return img
    
    def generate_gradcam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for specific target class.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index
        
        Returns:
            Grad-CAM heatmap (H, W) in range [0, 1]
        """
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=image, targets=targets)
        
        # Return first image's CAM (we only have one)
        return grayscale_cam[0, :]
    
    def visualize_prediction(self, image: torch.Tensor, true_label: int,
                            pred_label: int, confidence: float,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize prediction with Grad-CAM for both true and predicted classes.
        
        Args:
            image: Input image tensor (1, C, H, W)
            true_label: True class index
            pred_label: Predicted class index
            confidence: Prediction confidence
            save_path: Path to save visualization
        
        Returns:
            Matplotlib figure
        """
        # Denormalize image
        rgb_img = self.denormalize_image(image.squeeze(0))
        
        # Generate Grad-CAMs
        cam_true = self.generate_gradcam(image, true_label)
        cam_pred = self.generate_gradcam(image, pred_label)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title('Original Image', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM for true class
        cam_true_img = show_cam_on_image(rgb_img, cam_true, use_rgb=True)
        axes[1].imshow(cam_true_img)
        axes[1].set_title(f'True Class Focus\n{self.class_names[true_label]}', 
                         fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        # Grad-CAM for predicted class
        cam_pred_img = show_cam_on_image(rgb_img, cam_pred, use_rgb=True)
        axes[2].imshow(cam_pred_img)
        axes[2].set_title(f'Predicted Class Focus\n{self.class_names[pred_label]} ({confidence:.1f}%)', 
                         fontsize=10, fontweight='bold')
        axes[2].axis('off')
        
        # Overlay comparison
        axes[3].imshow(rgb_img, alpha=0.5)
        axes[3].imshow(cam_pred, alpha=0.5, cmap='jet')
        axes[3].set_title('Prediction Overlay', fontsize=10, fontweight='bold')
        axes[3].axis('off')
        
        correct = "CORRECT" if true_label == pred_label else "INCORRECT"
        fig.suptitle(f'Prediction: {correct}', fontsize=12, fontweight='bold',
                    color='green' if correct == "CORRECT" else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return fig
    
    def analyze_model(self, data_loader: DataLoader, num_samples: int = 100,
                     output_dir: str = './gradcam_results') -> Dict[str, any]:
        """
        Analyze model predictions with Grad-CAM.
        
        Args:
            data_loader: Validation data loader
            num_samples: Number of samples to analyze
            output_dir: Directory to save results
        
        Returns:
            Dictionary with analysis results
        """
        self.model.eval()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        correct_predictions = []
        incorrect_predictions = []
        
        print(f"Analyzing {num_samples} samples with Grad-CAM...")
        
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                if sample_count >= num_samples:
                    break
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probs = torch.exp(outputs)  # Convert log_softmax to probabilities
                confidences, predictions = torch.max(probs, 1)
                
                # Analyze each image in batch
                for i in range(images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    img = images[i:i+1]
                    true_label = labels[i].item()
                    pred_label = predictions[i].item()
                    confidence = confidences[i].item() * 100
                    
                    result = {
                        'image': img,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence,
                        'correct': true_label == pred_label
                    }
                    
                    if result['correct']:
                        correct_predictions.append(result)
                    else:
                        incorrect_predictions.append(result)
                    
                    sample_count += 1
        
        print(f"\nAnalyzed {sample_count} samples:")
        print(f"  Correct: {len(correct_predictions)}")
        print(f"  Incorrect: {len(incorrect_predictions)}")
        
        # Save worst predictions with Grad-CAM
        worst_dir = Path(output_dir) / 'worst_predictions'
        worst_dir.mkdir(exist_ok=True)
        
        # Sort by confidence (high confidence errors are most interesting)
        incorrect_sorted = sorted(incorrect_predictions, key=lambda x: x['confidence'], reverse=True)
        
        for idx, result in enumerate(incorrect_sorted[:20]):  # Top 20 worst
            save_path = worst_dir / f'error_{idx+1}_conf{result["confidence"]:.0f}.png'
            self.visualize_prediction(
                result['image'],
                result['true_label'],
                result['pred_label'],
                result['confidence'],
                str(save_path)
            )
        
        print(f"\nSaved visualizations to: {output_dir}")
        
        return {
            'total_analyzed': sample_count,
            'correct': len(correct_predictions),
            'incorrect': len(incorrect_predictions),
            'accuracy': 100.0 * len(correct_predictions) / sample_count
        }

