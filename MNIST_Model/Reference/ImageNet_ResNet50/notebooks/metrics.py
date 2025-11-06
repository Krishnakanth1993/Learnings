"""
Classification Metrics Module for ImageNet Classifier
Handles loading and display of pre-computed classification metrics.

This module provides:
- load_metrics(): Load metrics from JSON file
- get_class_metrics(): Get metrics for specific classes
- format_metrics_table(): Format metrics for display
- compute_metrics_from_validation(): Compute metrics from validation dataset
- Sample metrics data structure

Author: Krishnakanth
Date: 2025-10-26
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score
import numpy as np


@dataclass
class ClassMetrics:
    """Data class for individual class metrics."""
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class ModelMetrics:
    """Data class for overall model metrics."""
    accuracy: float
    top5_accuracy: float
    macro_avg_precision: float
    macro_avg_recall: float
    macro_avg_f1: float
    weighted_avg_precision: float
    weighted_avg_recall: float
    weighted_avg_f1: float


class MetricsManager:
    """Manages loading and formatting of classification metrics."""
    
    def __init__(self, metrics_file: str = "metrics.json"):
        """
        Initialize metrics manager.
        
        Args:
            metrics_file: Path to the metrics JSON file
        """
        self.metrics_file = metrics_file
        self.class_metrics: Dict[str, ClassMetrics] = {}
        self.model_metrics: Optional[ModelMetrics] = None
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from JSON file."""
        if not os.path.exists(self.metrics_file):
            print(f"Warning: Metrics file '{self.metrics_file}' not found. Using sample data.")
            self._create_sample_metrics()
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            # Load class metrics
            if 'class_metrics' in data:
                for class_name, metrics in data['class_metrics'].items():
                    self.class_metrics[class_name] = ClassMetrics(
                        precision=metrics.get('precision', 0.0),
                        recall=metrics.get('recall', 0.0),
                        f1_score=metrics.get('f1_score', 0.0),
                        support=metrics.get('support', 0)
                    )
            
            # Load model metrics
            if 'model_metrics' in data:
                model_data = data['model_metrics']
                self.model_metrics = ModelMetrics(
                    accuracy=model_data.get('accuracy', 0.0),
                    top5_accuracy=model_data.get('top5_accuracy', 0.0),
                    macro_avg_precision=model_data.get('macro_avg_precision', 0.0),
                    macro_avg_recall=model_data.get('macro_avg_recall', 0.0),
                    macro_avg_f1=model_data.get('macro_avg_f1', 0.0),
                    weighted_avg_precision=model_data.get('weighted_avg_precision', 0.0),
                    weighted_avg_recall=model_data.get('weighted_avg_recall', 0.0),
                    weighted_avg_f1=model_data.get('weighted_avg_f1', 0.0)
                )
            
            print(f" Loaded metrics for {len(self.class_metrics)} classes")
            
        except Exception as e:
            print(f"Error loading metrics: {e}")
            self._create_sample_metrics()
    
    def _create_sample_metrics(self):
        """Create sample metrics data for demonstration."""
        # Sample class metrics (these would be replaced with real validation results)
        sample_classes = [
            'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
            'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
            'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin',
            'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite',
            'bald_eagle', 'vulture', 'great_grey_owl', 'lion', 'tiger', 'jaguar',
            'leopard', 'snow_leopard', 'lynx', 'bobcat', 'clouded_leopard',
            'sunda_clouded_leopard', 'cheetah', 'brown_bear', 'american_black_bear',
            'ice_bear', 'sloth_bear', 'aircraft_carrier', 'airliner', 'airship',
            'ambulance', 'bicycle_built_for_two', 'bobsled', 'bullet_train', 'cab',
            'canoe', 'car_mirror', 'carousel', 'car_wheel', 'catamaran',
            'container_ship', 'convertible', 'crane', 'dogsled', 'electric_locomotive',
            'freight_car', 'go_kart', 'golfcart', 'gondola', 'garbage_truck',
            'half_track', 'harvester', 'horse_cart', 'jeep', 'limousine', 'liner',
            'minibus', 'minivan', 'moped', 'motor_scooter', 'mountain_bike',
            'moving_van', 'oxcart', 'passenger_car', 'pickup', 'police_van',
            'racer', 'recreational_vehicle', 'school_bus', 'schooner', 'snowmobile',
            'snowplow', 'space_shuttle', 'speedboat', 'sports_car', 'steam_locomotive',
            'streetcar', 'submarine', 'tow_truck', 'tractor', 'trailer_truck',
            'tricycle', 'trimaran', 'trolleybus', 'warplane', 'yawl'
        ]
        
        import random
        random.seed(42)  # For reproducible sample data
        
        for class_name in sample_classes:
            self.class_metrics[class_name] = ClassMetrics(
                precision=round(random.uniform(0.6, 0.95), 3),
                recall=round(random.uniform(0.5, 0.9), 3),
                f1_score=round(random.uniform(0.55, 0.92), 3),
                support=random.randint(50, 500)
            )
        
        # Sample model metrics
        self.model_metrics = ModelMetrics(
            accuracy=0.8234,
            top5_accuracy=0.9456,
            macro_avg_precision=0.7891,
            macro_avg_recall=0.7654,
            macro_avg_f1=0.7767,
            weighted_avg_precision=0.8234,
            weighted_avg_recall=0.8234,
            weighted_avg_f1=0.8234
        )
        
        print(" Created sample metrics data")
    
    def get_class_metrics(self, class_name: str) -> Optional[ClassMetrics]:
        """
        Get metrics for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            ClassMetrics object or None if not found
        """
        return self.class_metrics.get(class_name)
    
    def get_top_classes_metrics(self, class_names: List[str]) -> List[Tuple[str, ClassMetrics]]:
        """
        Get metrics for a list of classes.
        
        Args:
            class_names: List of class names
            
        Returns:
            List of tuples (class_name, ClassMetrics)
        """
        results = []
        for class_name in class_names:
            metrics = self.get_class_metrics(class_name)
            if metrics:
                results.append((class_name, metrics))
            else:
                # Create default metrics if not found
                default_metrics = ClassMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    support=0
                )
                results.append((class_name, default_metrics))
        
        return results
    
    def format_metrics_table(self, class_names: List[str]) -> str:
        """
        Format metrics as HTML table for display.
        
        Args:
            class_names: List of class names to display
            
        Returns:
            HTML formatted table string
        """
        if not class_names:
            return "<p>No classes to display</p>"
        
        # Get metrics for the classes
        class_metrics = self.get_top_classes_metrics(class_names)
        
        # Create HTML table with theme-consistent background
        html = """
        <div style="margin: 20px 0; padding: 15px; background-color: #f0f2f6; border-radius: 8px;">
            <h3 style="color: #1f77b4; margin-bottom: 15px;"> Classification Metrics (Top-5 Classes)</h3>
            <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; background-color: #ffffff; border-radius: 5px; overflow: hidden;">
                <thead>
                    <tr style="background-color: #1f77b4;">
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: left; color: #ffffff;">Class</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: center; color: #ffffff;">Precision</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: center; color: #ffffff;">Recall</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: center; color: #ffffff;">F1-Score</th>
                        <th style="padding: 12px; border: 1px solid #ddd; text-align: center; color: #ffffff;">Support</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, (class_name, metrics) in enumerate(class_metrics):
            # Alternate row colors with theme-consistent colors
            row_color = "#f0f2f6" if i % 2 == 0 else "#ffffff"
            
            # Format class name
            formatted_name = class_name.replace('_', ' ').title()
            
            # Color code metrics based on performance
            def get_color(value):
                if value >= 0.8:
                    return "#28a745"  # Green
                elif value >= 0.6:
                    return "#ffc107"  # Yellow
                else:
                    return "#dc3545"  # Red
            
            precision_color = get_color(metrics.precision)
            recall_color = get_color(metrics.recall)
            f1_color = get_color(metrics.f1_score)
            
            html += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{formatted_name}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: {precision_color}; font-weight: bold;">{metrics.precision:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: {recall_color}; font-weight: bold;">{metrics.recall:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: {f1_color}; font-weight: bold;">{metrics.f1_score:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{metrics.support}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def format_model_summary(self) -> str:
        """
        Format overall model metrics summary.
        
        Returns:
            HTML formatted summary string
        """
        if not self.model_metrics:
            return "<p>Model metrics not available</p>"
        
        metrics = self.model_metrics
        
        html = f"""
        <div style="margin: 20px 0; padding: 15px; background-color: #f0f2f6; border-radius: 8px;">
            <h3 style="color: #1f77b4; margin-bottom: 15px;"> Model Performance Summary</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="text-align: center; padding: 10px; background-color: #ffffff; border-radius: 5px; border: 1px solid #ddd;">
                    <h4 style="margin: 0; color: #28a745;">Top-1 Accuracy</h4>
                    <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0; color: #28a745;">{metrics.accuracy:.1%}</p>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #ffffff; border-radius: 5px; border: 1px solid #ddd;">
                    <h4 style="margin: 0; color: #17a2b8;">Top-5 Accuracy</h4>
                    <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0; color: #17a2b8;">{metrics.top5_accuracy:.1%}</p>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #ffffff; border-radius: 5px; border: 1px solid #ddd;">
                    <h4 style="margin: 0; color: #6f42c1;">Macro F1-Score</h4>
                    <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0; color: #6f42c1;">{metrics.macro_avg_f1:.3f}</p>
                </div>
                <div style="text-align: center; padding: 10px; background-color: #ffffff; border-radius: 5px; border: 1px solid #ddd;">
                    <h4 style="margin: 0; color: #fd7e14;">Weighted F1-Score</h4>
                    <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0; color: #fd7e14;">{metrics.weighted_avg_f1:.3f}</p>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def save_sample_metrics(self, output_file: str = "metrics.json"):
        """
        Save sample metrics to JSON file.
        
        Args:
            output_file: Path to output file
        """
        data = {
            "class_metrics": {},
            "model_metrics": {}
        }
        
        # Save class metrics
        for class_name, metrics in self.class_metrics.items():
            data["class_metrics"][class_name] = {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "support": metrics.support
            }
        
        # Save model metrics
        if self.model_metrics:
            data["model_metrics"] = {
                "accuracy": self.model_metrics.accuracy,
                "top5_accuracy": self.model_metrics.top5_accuracy,
                "macro_avg_precision": self.model_metrics.macro_avg_precision,
                "macro_avg_recall": self.model_metrics.macro_avg_recall,
                "macro_avg_f1": self.model_metrics.macro_avg_f1,
                "weighted_avg_precision": self.model_metrics.weighted_avg_precision,
                "weighted_avg_recall": self.model_metrics.weighted_avg_recall,
                "weighted_avg_f1": self.model_metrics.weighted_avg_f1
            }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Sample metrics saved to {output_file}")


def compute_metrics_from_validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    output_file: str = "metrics.json",
    num_classes: int = 1000
) -> MetricsManager:
    """
    Compute precision, recall, and F1 scores from a validation dataset.
    
    Args:
        model: Trained PyTorch model
        val_loader: DataLoader for validation dataset
        device: Device to run inference on
        class_names: List of class names (ImageNet class names)
        output_file: Path to save metrics JSON file
        num_classes: Number of classes (default: 1000 for ImageNet)
        
    Returns:
        MetricsManager instance with computed metrics
    """
    print("Computing metrics from validation dataset...")
    print(f"  Device: {device}")
    print(f"  Number of classes: {num_classes}")
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            probs = torch.exp(outputs)  # Convert log probabilities to probabilities
            
            # Get predicted class indices
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f"  Total samples: {len(all_labels)}")
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Compute overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Compute top-5 accuracy
    top5_accuracy = top_k_accuracy_score(all_labels, all_probs, k=5)
    
    # Compute macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Compute weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Create class metrics dictionary
    class_metrics_dict = {}
    for i in range(num_classes):
        if i < len(class_names):
            class_name = class_names[i]
        else:
            class_name = f"class_{i}"
        
        class_metrics_dict[class_name] = {
            "precision": float(precision[i]) if i < len(precision) else 0.0,
            "recall": float(recall[i]) if i < len(recall) else 0.0,
            "f1_score": float(f1[i]) if i < len(f1) else 0.0,
            "support": int(support[i]) if i < len(support) else 0
        }
    
    # Create model metrics
    model_metrics_dict = {
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_avg_precision": float(macro_precision),
        "macro_avg_recall": float(macro_recall),
        "macro_avg_f1": float(macro_f1),
        "weighted_avg_precision": float(weighted_precision),
        "weighted_avg_recall": float(weighted_recall),
        "weighted_avg_f1": float(weighted_f1)
    }
    
    # Save to JSON file
    metrics_data = {
        "class_metrics": class_metrics_dict,
        "model_metrics": model_metrics_dict
    }
    
    with open(output_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"  Metrics computed and saved to {output_file}")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    
    # Create and return MetricsManager with computed metrics
    metrics_manager = MetricsManager(output_file)
    return metrics_manager


# Convenience functions
def load_metrics(metrics_file: str = "metrics.json") -> MetricsManager:
    """
    Load metrics from file.
    
    Args:
        metrics_file: Path to metrics file
        
    Returns:
        MetricsManager instance
    """
    return MetricsManager(metrics_file)


def get_class_metrics(class_name: str, metrics_manager: MetricsManager) -> Optional[ClassMetrics]:
    """
    Get metrics for a specific class.
    
    Args:
        class_name: Name of the class
        metrics_manager: MetricsManager instance
        
    Returns:
        ClassMetrics object or None
    """
    return metrics_manager.get_class_metrics(class_name)


def format_metrics_table(class_names: List[str], metrics_manager: MetricsManager) -> str:
    """
    Format metrics table for display.
    
    Args:
        class_names: List of class names
        metrics_manager: MetricsManager instance
        
    Returns:
        HTML formatted table
    """
    return metrics_manager.format_metrics_table(class_names)


# Example usage
if __name__ == "__main__":
    # Create and test metrics manager
    metrics_manager = MetricsManager()
    
    # Test with sample classes
    sample_classes = ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead']
    
    # Format metrics table
    table_html = metrics_manager.format_metrics_table(sample_classes)
    print("Metrics Table HTML:")
    print(table_html)
    
    # Format model summary
    summary_html = metrics_manager.format_model_summary()
    print("\nModel Summary HTML:")
    print(summary_html)
    
    # Save sample metrics
    metrics_manager.save_sample_metrics()
