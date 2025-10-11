"""
CIFAR-100 Image Classification App
Deployed on Hugging Face Spaces with Gradio

Author: Krishnakanth
Date: 2025-10-10
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List
import torchvision.transforms as transforms
import plotly.graph_objects as go
import sys
from dataclasses import dataclass
import types

# Import model architecture
from model import CIFAR100ResNet34, ModelConfig

# Create dummy TrainingConfig class for unpickling checkpoints
@dataclass
class TrainingConfig:
    """Dummy class to allow unpickling checkpoints saved with TrainingConfig."""
    epochs: int = 100
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    seed: int = 1
    optimizer_type: str = 'Adam'
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-08
    rmsprop_alpha: float = 0.99
    scheduler_type: str = 'OneCycleLR'
    cosine_t_max: int = 20
    exponential_gamma: float = 0.95
    plateau_mode: str = 'min'
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 0.0001
    onecycle_max_lr: float = 0.003
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 5
    onecycle_final_div_factor: float = 1000.0
    onecycle_anneal_strategy: str = 'cos'

# Register dummy module for unpickling
cifar100_training_module = types.ModuleType('cifar100_training')
cifar100_training_module.TrainingConfig = TrainingConfig
sys.modules['cifar100_training'] = cifar100_training_module

# Add to safe globals for torch.load
try:
    torch.serialization.add_safe_globals([TrainingConfig])
except:
    pass  # Older PyTorch versions don't have this

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

# CIFAR-100 normalization values
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Global variables for model
model = None
device = None


def load_model(model_path: str = "cifar100_model.pth"):
    """Load the trained CIFAR-100 model."""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model configuration
    config = ModelConfig(
        input_channels=3,
        input_size=(32, 32),
        num_classes=100,
        dropout_rate=0.0  # Model was trained with no dropout
    )
    
    # Initialize model
    model = CIFAR100ResNet34(config)
    
    # Load trained weights
    try:
        # Load checkpoint (weights_only=False needed for PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded with metrics: {checkpoint.get('metrics', {})}")
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully on {device}")
        print(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    transform = get_transform()
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def predict(image: Image.Image) -> Tuple[Dict[str, float], str, str]:
    """
    Make prediction on image.
    
    Returns:
        - Dictionary of top predictions {class: probability}
        - HTML formatted main prediction
        - Plotly chart (not used in Gradio, for reference)
    """
    if model is None:
        return {}, "❌ Model not loaded", ""
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            
            # Get model output (log probabilities)
            output = model(image_tensor)
            
            # Convert to probabilities
            probabilities = torch.exp(output)
            
            # Get top-10 predictions
            top_probs, top_indices = torch.topk(probabilities, 10, dim=1)
            
            top_probs = top_probs[0].cpu().numpy()
            top_indices = top_indices[0].cpu().numpy()
        
        # Get predicted class
        predicted_class = CIFAR100_CLASSES[top_indices[0]]
        confidence = top_probs[0]
        
        # Create results dictionary for Gradio Label output
        results_dict = {}
        for idx, prob in zip(top_indices, top_probs):
            class_name = CIFAR100_CLASSES[idx].replace('_', ' ').title()
            results_dict[class_name] = float(prob)
        
        # Create formatted output
        confidence_pct = confidence * 100
        
        if confidence_pct > 70:
            conf_emoji = "✅"
            conf_text = "High Confidence"
            color = "#28a745"
        elif confidence_pct > 40:
            conf_emoji = "⚠️"
            conf_text = "Medium Confidence"
            color = "#ffc107"
        else:
            conf_emoji = "❌"
            conf_text = "Low Confidence"
            color = "#dc3545"
        
        main_prediction = f"""
        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
            <h2 style='color: #1f77b4; margin: 10px 0;'>Predicted Class</h2>
            <h1 style='color: #1f77b4; font-size: 2.5em; margin: 15px 0;'>
                {predicted_class.replace('_', ' ').upper()}
            </h1>
            <h2 style='color: {color}; margin: 10px 0;'>
                {conf_emoji} {confidence_pct:.2f}%
            </h2>
            <p style='color: #666; font-size: 1.2em;'>{conf_text}</p>
        </div>
        """
        
        return results_dict, main_prediction, ""
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        return {}, error_msg, ""


def create_interface():
    """Create Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #1f77b4, #9467bd) !important;
        border: none !important;
    }
    footer {
        visibility: hidden;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, title="CIFAR-100 Classifier", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🖼️ CIFAR-100 Image Classifier
        
        Upload an image and the AI will classify it into one of **100 different categories** with confidence scores.
        Built with **PyTorch ResNet-34** architecture.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📤 Upload Image")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload an image",
                    sources=["upload", "clipboard", "webcam"],
                    height=400
                )
                
                predict_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 💡 Tips for Best Results
                - Use clear, well-lit images
                - Center the main object
                - Any size works (auto-resized to 32×32)
                - Supported: JPG, PNG, BMP, WEBP
                """)
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## 🎯 Classification Results")
                
                main_output = gr.HTML(label="Main Prediction")
                
                gr.Markdown("### 📊 Top 10 Predictions")
                
                label_output = gr.Label(
                    num_top_classes=10,
                    label="Confidence Scores",
                    show_label=False
                )
        
        # Additional info section
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🤖 Model Information
                
                - **Architecture**: ResNet-34 with Bottleneck Layers
                - **Parameters**: ~21 Million
                - **Dataset**: CIFAR-100 (60,000 images, 100 classes)
                - **Input Size**: 32×32 RGB images
                - **Categories**: Animals, vehicles, household items, nature scenes, and more
                """)
            
            with gr.Column():
                gr.Markdown("""
                ### 📚 Sample Categories
                
                **Animals**: bear, dolphin, elephant, fox, leopard, tiger, whale  
                **Vehicles**: bicycle, bus, motorcycle, train, tractor  
                **Nature**: cloud, forest, mountain, sea, plain  
                **Objects**: chair, clock, lamp, telephone, keyboard  
                **Plants**: maple_tree, oak_tree, orchid, rose, sunflower  
                
                *...and 75 more categories!*
                """)
        
        # Examples
        gr.Markdown("### 🎨 Try These Examples")
        gr.Examples(
            examples=["sample_images/apple_s_000028.png",
        "sample_images/breakfast_table_s_000178.png",
        "sample_images/cichlid_fish_s_000888.png"],
        inputs=image_input,
        label="Example Images")
        
        # Footer
        gr.Markdown("""
        ---
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p style='font-size: 1.1em;'>Built with ❤️ using <strong>PyTorch</strong>, <strong>Gradio</strong>, and <strong>Hugging Face Spaces</strong></p>
            <p>Model: ResNet-34 trained on CIFAR-100 dataset</p>
            <p style='font-size: 0.9em; margin-top: 10px;'>Created by Krishnakanth | © 2025</p>
        </div>
        """)
        
        # Connect button to prediction function
        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, main_output, gr.Textbox(visible=False)]
        )
        
        # Also trigger on image upload
        image_input.change(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, main_output, gr.Textbox(visible=False)]
        )
    
    return demo


def main():
    """Main function to run the Gradio app."""
    print("="*60)
    print("🖼️  CIFAR-100 Image Classifier")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    success = load_model("cifar100_model.pth")
    
    if not success:
        print("❌ Failed to load model. Please check if cifar100_model.pth exists.")
        return
    
    print("\n🚀 Creating Gradio interface...")
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n✅ Interface created successfully!")
    print("="*60)
    
    # Launch app
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
