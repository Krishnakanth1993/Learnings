---
title: CIFAR-100 Image Classifier
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# 🖼️ CIFAR-100 Image Classifier

An interactive deep learning application that classifies images into 100 different categories using a ResNet-34 architecture.

## 🌟 Features

- **🎯 Accurate Classification**: Classify images into 100 different categories
- **📊 Top-10 Predictions**: View multiple predictions with confidence scores
- **📈 Real-time Visualization**: Confidence bars and probability distribution
- **🔥 Fast Inference**: Optimized for quick predictions
- **📱 Mobile Friendly**: Works on any device
- **🎨 Beautiful UI**: Clean and intuitive Gradio interface

## 🤖 Model Details

- **Architecture**: ResNet-34 with BasicBlock
- **Framework**: PyTorch
- **Dataset**: CIFAR-100 (60,000 32×32 color images)
- **Input Size**: 32×32 RGB images (auto-resized)
- **Parameters**: ~21 Million
- **Training Features**:
  - Data augmentation (Albumentations)
  - Dropout regularization (5%)
  - Learning rate scheduling
  - Batch normalization

## 📚 100 Classes

The model can classify images into these categories:

### 🐾 Animals (47 classes)
bear, beaver, bee, beetle, butterfly, camel, caterpillar, cattle, chimpanzee, cockroach, crab, crocodile, dinosaur, dolphin, elephant, flatfish, fox, hamster, kangaroo, leopard, lion, lizard, lobster, mouse, otter, porcupine, possum, rabbit, raccoon, ray, seal, shark, shrew, skunk, snail, snake, spider, squirrel, tiger, trout, turtle, whale, wolf, worm

### 🚗 Vehicles (8 classes)
bicycle, bus, motorcycle, pickup_truck, streetcar, tank, tractor, train

### 🏠 Household Items (15 classes)
bed, bottle, bowl, can, chair, clock, couch, cup, keyboard, lamp, plate, table, telephone, television, wardrobe

### 🌳 Nature & Places (10 classes)
bridge, castle, cloud, forest, house, mountain, plain, road, sea, skyscraper

### 🌺 Plants & Flowers (9 classes)
maple_tree, oak_tree, orchid, palm_tree, pine_tree, poppy, rose, sunflower, tulip, willow_tree

### 👥 People (5 classes)
baby, boy, girl, man, woman

### 🍎 Food & Others (6 classes)
apple, aquarium_fish, lawn_mower, mushroom, orange, pear, rocket, sweet_pepper

## 🚀 How to Use

1. **Upload Image**: Click the upload area or drag & drop an image
2. **Automatic Classification**: The model classifies automatically
3. **View Results**: See the predicted class with confidence score
4. **Explore Predictions**: Check top-10 predictions with probabilities
5. **Try Examples**: Use sample images (if available)

## 💡 Tips for Best Results

- Use clear, well-lit images
- Center the main object in the frame
- Avoid heavily filtered or edited images
- Any image size works (automatically resized)
- Best results with images similar to CIFAR-100 training data

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.0+
- **Interface**: Gradio 4.0+
- **Deployment**: Hugging Face Spaces
- **Image Processing**: PIL, torchvision

## 📖 Model Architecture

```
ResNet-34 Structure:
┌──────────────────────────────────┐
│ Input: 32×32 RGB Image           │
├──────────────────────────────────┤
│ Conv1: 3×3, 64 channels          │
│ BatchNorm + ReLU                 │
├──────────────────────────────────┤
│ Layer 1: 3 BasicBlocks (64ch)    │
│ Layer 2: 4 BasicBlocks (128ch)   │
│ Layer 3: 6 BasicBlocks (256ch)   │
│ Layer 4: 3 BasicBlocks (512ch)   │
├──────────────────────────────────┤
│ Global Average Pooling           │
│ Dropout (5%)                     │
│ Fully Connected: 512 → 100       │
│ Log Softmax                      │
├──────────────────────────────────┤
│ Output: 100 class probabilities  │
└──────────────────────────────────┘
```

## 🎓 Training Details

The model was trained using:
- **Optimizer**: Adam with weight decay
- **Learning Rate**: OneCycleLR scheduler
- **Regularization**: Dropout (5%), Weight Decay (0.0001)
- **Augmentation**: HorizontalFlip, ShiftScaleRotate, CoarseDropout
- **Batch Size**: 128-256
- **Epochs**: 100+
- **Framework**: PyTorch with Albumentations

## 📊 Performance

- **Inference Time**: 50-200ms (CPU), 10-30ms (GPU)
- **Accuracy**: Trained with advanced techniques
- **Reliability**: Consistent predictions across similar images

## 🎯 Use Cases

- **Educational**: Learn about image classification
- **Research**: Test model performance on new images
- **Fun**: See what the AI thinks about your photos
- **Prototyping**: Base for custom applications

## 👨‍💻 Author

**Krishnakanth**

Developed as part of deep learning experimentation with CIFAR-100 dataset.

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- **CIFAR-100 Dataset**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **ResNet Architecture**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Hugging Face**: For the amazing Spaces platform
- **Gradio**: For the beautiful interface framework

## 🔗 Links

- [CIFAR-100 Dataset Info](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Gradio Documentation](https://www.gradio.app/docs/)

---

**Try it now! Upload an image and see the magic of deep learning! ✨**
