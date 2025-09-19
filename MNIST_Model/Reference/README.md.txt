# 🧠 CNN Classifier for MNIST Handwritten Digits

This project implements a **compact Convolutional Neural Network (CNN)** with approximately **20,000 parameters**, designed and trained on the **MNIST** dataset using **PyTorch**.

## 📚 1. About the MNIST Dataset

**MNIST** (Modified National Institute of Standards and Technology) is a widely used dataset for benchmarking image classification models.

- **Images:** 70,000 grayscale images of handwritten digits (0–9)
- **Size:** 28x28 pixels
- **Split:**
  - 60,000 training images
  - 10,000 test images
- **Classes:** 10 (digits 0 through 9)

---

## 🧠 4. CNN Architecture & Parameters

### 📊 **Parameter Breakdown**

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Conv2d-1 | Conv2d | [-1, 14, 28, 28] | 140 |
| BatchNorm2d-2 | BatchNorm2d | [-1, 14, 28, 28] | 28 |
| Conv2d-3 | Conv2d | [-1, 14, 28, 28] | 1,778 |
| BatchNorm2d-4 | BatchNorm2d | [-1, 14, 28, 28] | 28 |
| Conv2d-5 | Conv2d | [-1, 14, 28, 28] | 1,778 |
| BatchNorm2d-6 | BatchNorm2d | [-1, 14, 28, 28] | 28 |
| Dropout-7 | Dropout | [-1, 14, 14, 14] | 0 |
| Conv2d-8 | Conv2d | [-1, 24, 14, 14] | 3,048 |
| BatchNorm2d-9 | BatchNorm2d | [-1, 24, 14, 14] | 48 |
| Conv2d-10 | Conv2d | [-1, 24, 14, 14] | 5,208 |
| BatchNorm2d-11 | BatchNorm2d | [-1, 24, 14, 14] | 48 |
| Dropout-12 | Dropout | [-1, 24, 7, 7] | 0 |
| Conv2d-13 | Conv2d | [-1, 16, 7, 7] | 3,472 |
| BatchNorm2d-14 | BatchNorm2d | [-1, 16, 7, 7] | 32 |
| Conv2d-15 | Conv2d | [-1, 16, 7, 7] | 2,320 |
| BatchNorm2d-16 | BatchNorm2d | [-1, 16, 7, 7] | 32 |
| Dropout-17 | Dropout | [-1, 16] | 0 |
| Linear-18 | Linear | [-1, 50] | 850 |
| Linear-19 | Linear | [-1, 10] | 510 |

**Total Parameters: 19,348**

### 🏗️ **Architecture Features**
- **Batch Normalization:** Yes
- **Dropout:** Yes (p=0.15)
- **Global Average Pooling:** Yes
- **Fully Connected Layers:** Yes

---

## �� 5. Training Results

### �� **Best Performance (Epoch 14)**
- **Training Accuracy:** 99.17%
- **Validation Accuracy:** 99.38% (9,938/10,000)
- **Test Accuracy:** 99.49% (9,949/10,000)
- **Training Loss:** 0.0274
- **Validation Loss:** 0.0216
- **Test Loss:** 0.0202

The model achieved **99.4%+ accuracy** on all datasets by epoch 14, demonstrating excellent performance with a compact architecture of only ~19K parameters.

---

## 📊 6. Experiment Log

| Ex_Number | CommitID | Link | Model Details | Accuracy (Epoch 1) | Parameters | Changes | Observations |
|-----------|----------|------|---------------|-------------------|------------|---------|--------------|
| 1 | 6c122e9096f773f201c64d0bdfd183cdb7ceb506 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/6c122e9096f773f201c64d0bdfd183cdb7ceb506/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 12,026<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 1 Results**<br/>Train: Loss=0.8093, Accuracy=68.30%<br/>Test: Loss=0.3290, Accuracy=89.64% (8,964/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- No Padding<br/>- No Stride<br/>- 1 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Initial experiment | Accuracy is decent with 12K parameters. Since I have 8K more room, will try to increase the kernel size in first block. |
| 2 | 3cbc61ec60c447a2afe634c9e6035a60eb00be56 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/3cbc61ec60c447a2afe634c9e6035a60eb00be56/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 12,026<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 1 Results**<br/>Train: Loss=0.3290, Accuracy=67.69%<br/>Test: Loss=0.3839, Accuracy=88.19% (8,819/10,000) | **Configuration**<br/>- Kernel size: First block: 5x5, Second block: 3x3<br/>- No Padding<br/>- No Stride<br/>- 1 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Kernel size: First block: 5x5, Second block: 3x3 | Increasing Kernel Size in first block to 5x5 led to no increase in accuracy (rather slight dip). Also, it led to drastic increment in RF and reduction in feature map sizes in second block leading to reduced parameters in FC. |
| 3 | 903258dc9b5ebc7f2ff6835425460a91abccf487 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/903258dc9b5ebc7f2ff6835425460a91abccf487/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 35,946<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 1 Results**<br/>Train: Loss=0.1876, Accuracy=79.20%<br/>Test: Loss=0.1651, Accuracy=94.75% (9,475/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- No Padding<br/>- No Stride<br/>- 1 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Channel increase: Block 1: 10→16, Block 2: 16→32 | Accuracy increased as number of parameters tripled. Also, architecture is expanding in nature contrary to leading papers. Even logically, no. of edges > shapes. Trying to swap block 1 and 2 in next experiment. |
| 4 | 9e8681263eedc007bbb3f1583a2bd0179a6c6c9b | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/9e8681263eedc007bbb3f1583a2bd0179a6c6c9b/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 32,090<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 1 Results**<br/>Train: Loss=0.1228, Accuracy=78.24%<br/>Test: Loss=0.1269, Accuracy=96.11% (9,611/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- No Padding<br/>- No Stride<br/>- 1 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Channel increase: Block 1: 16→32, Block 2: 32→16 | Accuracy improved a bit. Parameters has come down to 32K. Will experiment with adding padding to keep the feature maps constant in size. |
| 5 | 29ddd083928a9fc52b5aa0b9082c447e8ea11fd1 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/29ddd083928a9fc52b5aa0b9082c447e8ea11fd1/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 38,330<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 10 Results**<br/>Train: Loss=0.0563, Accuracy=97.72%<br/>Test: Loss=0.0422, Accuracy=98.63% (9,863/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding: First block: 1, Second block: no padding<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Padding: First block: 1, Second block: no padding | Accuracy improved with padding. Parameter count increased to 38K. Try to add another Max pool layer to reduce parameters. Hoping Accuracy sustains. |
| 6 | 5595d50a44953e4173c0aff939bb4a63f8dd3c06 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/5595d50a44953e4173c0aff939bb4a63f8dd3c06/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,186<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 10 Results**<br/>Train: Loss=0.0854, Accuracy=96.97%<br/>Test: Loss=0.0638, Accuracy=98.04% (9,804/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Padding in all blocks, Block increased: CCMCCMCC(FC) | Parameters drastically reduced to 22K. Accuracy has dropped by 0.55%. |
| 7 | 4d73d40b6861ce3ba03c7a698f8c0e9eaff0b974 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/4d73d40b6861ce3ba03c7a698f8c0e9eaff0b974/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,186<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 10 Results**<br/>Train: Loss=0.0540, Accuracy=98.06%<br/>Test: Loss=0.0351, Accuracy=98.72% (9,872/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Scheduler step_size=1 → 5 | Accuracy improved by 0.68% by increasing the scheduler step size. |
| 8 | 52217f55963322fb389af124a7804b087dda3f3f | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/52217f55963322fb389af124a7804b087dda3f3f/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,186<br/>- Use of Batch Normalization: No<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 19 Results**<br/>Train: Loss=0.0106, Accuracy=98.45%<br/>Test: Loss=0.0273, Accuracy=99.17% (9,917/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Scheduler step_size=5 → 10 | Accuracy improved by 0.49% by increasing the scheduler step size. Planning to add Batch Normalization in next experiment. |
| 9 | 9bb76e1dadb5050199051265f9ae1d7d748e4881 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/9bb76e1dadb5050199051265f9ae1d7d748e4881/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,410<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 19 Results**<br/>Train: Loss=0.0274, Accuracy=99.26%<br/>Test: Loss=0.0180, Accuracy=99.46% (9,946/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 512 | Batch Normalization added after each conv layer | Accuracy of 99.46% reached. Number of parameters are marginally increased. Will attempt to increase batch size. |
| 10 | 342aebe3fe844c7831718ce422980fcebd1b785c | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/342aebe3fe844c7831718ce422980fcebd1b785c/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,186<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 19 Results**<br/>Train: Loss=0.0117, Accuracy=99.12%<br/>Test: Loss=0.0196, Accuracy=99.47% (9,947/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024 | Batch size doubled to 1024 | Accuracy is stable at 99.47%. Need to reduce number of parameters. |
| 11 | b387d71578be90595aff226bc852c78f99384a9f | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/b387d71578be90595aff226bc852c78f99384a9f/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,614<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 20 Results**<br/>Train: Loss=0.0435, Accuracy=99.09%<br/>Test: Loss=0.0216, Accuracy=99.28% (9,928/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024 | Channel decrease: Block 1: 32→28 | Parameter target met (<20k). But Accuracy reduced to 99.28% with reduction in channels in 1st block. To add Validation split in dataset and retry. |
| 12 | a1aa0e7c40debb981c283f20ff0c0fa5807b976e | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/a1aa0e7c40debb981c283f20ff0c0fa5807b976e/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,614<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 15 Results**<br/>Train: Loss=0.0489, Accuracy=99.00%<br/>Validation: Loss=0.0394, Accuracy=98.86% (9,886/10,000)<br/>Test: Loss=0.0214, Accuracy=99.35% (9,935/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024 | Added Validation split<br/>Train: 50,000 \| Validation: 10,000 \| Test: 10,000 | Final test accuracy improved. Will try adding dropout to check if model is overfitting. |
| 13 | 0cfeedc26fcea3bcc60f54354cf0ba2009a2a7ec | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/0cfeedc26fcea3bcc60f54354cf0ba2009a2a7ec/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 22,186<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: Yes<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 15 Results**<br/>Train: Loss=0.0829, Accuracy=98.33%<br/>Validation: Loss=0.0456, Accuracy=98.55% (9,855/10,000)<br/>Test: Loss=0.0252, Accuracy=99.23% (9,923/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024<br/>- Dropout: 0.5 | Dropout p=0.5 added | Both Test and Validation accuracy dropped after adding 50% dropout. Validation data had data augmentation applied while loading data. Test results are better than validation. |
| 14 | 6288cfa29c4a6e07b6e0972309186ec1fb51c4e7 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/6288cfa29c4a6e07b6e0972309186ec1fb51c4e7/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,614<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: Yes<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 20 Results**<br/>Train: Loss=0.0602, Accuracy=98.57%<br/>Validation: Loss=0.0320, Accuracy=98.99% (9,899/10,000)<br/>Test: Loss=0.0251, Accuracy=99.18% (9,918/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024<br/>- Dropout: 0.25 | Dropout reduced p=0.25<br/>Transformation on validation data made similar to that of test set | Validation accuracy improved. It's closer to test accuracy. Removing dropout in next iteration. |
| 15 | 90ab43e2cfa7956b462a001d9efe52d7ad361f08 | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/90ab43e2cfa7956b462a001d9efe52d7ad361f08/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,614<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: No<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 20 Results**<br/>Train: Loss=0.0070, Accuracy=99.84%<br/>Validation: Loss=0.0319, Accuracy=98.93% (9,893/10,000)<br/>Test: Loss=0.0259, Accuracy=99.19% (9,919/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024 | Dropout removed | Train accuracy too high. Validation is unable to improve after initial epochs. Adding back dropout. |
| 16 | bddc7e3f811dcdf003dc582611f11bb7baa4077f | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/bddc7e3f811dcdf003dc582611f11bb7baa4077f/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,614<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: Yes<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 20 Results**<br/>Train: Loss=0.0583, Accuracy=98.43%<br/>Validation: Loss=0.0336, Accuracy=98.90% (9,890/10,000)<br/>Test: Loss=0.0267, Accuracy=99.15% (9,915/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.001<br/>- Batch Size: 1024<br/>- Dropout: 0.3 | Dropout added: 0.3 | No major validation accuracy improvement. |
| 17 | 5e0f0b0c6ce0d11ba67ca66f85b390455abfaa6e | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/5e0f0b0c6ce0d11ba67ca66f85b390455abfaa6e/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,348<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: Yes<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 14 Results**<br/>Train: Loss=0.0274, Accuracy=99.17%<br/>Validation: Loss=0.0216, Accuracy=99.38% (9,938/10,000)<br/>Test: Loss=0.0202, Accuracy=99.49% (9,949/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.05<br/>- Batch Size: 1024<br/>- Dropout: 0.1 | LR to 0.05<br/>Adding dropout only in last block before FC<br/>Architecture change: GAP introduced with increased number of conv layers | Test accuracy > 99.4% achieved. Train accuracy is lesser than Validation accuracy. |
| 18 | 98349e99fdc56e086f59bc0cc1e0a08bbd6e84cd | [GitHub Link](https://github.com/Krishnakanth1993/Learnings/blob/98349e99fdc56e086f59bc0cc1e0a08bbd6e84cd/MNIST_Model/Reference/minimal_prameter_model_training.ipynb) | **Model Details**<br/>- Total Parameter Count: 19,348<br/>- Use of Batch Normalization: Yes<br/>- Use of Dropout: Yes<br/>- Use of a Fully Connected Layer or Global Average Pooling: Yes | **Epoch 19 Results**<br/>Train: Loss=0.0347, Accuracy=98.77%<br/>Validation: Loss=0.0240, Accuracy=99.30% (9,930/10,000)<br/>Test: Loss=0.0204, Accuracy=99.41% (9,941/10,000) | **Configuration**<br/>- Kernel size: 3x3<br/>- Padding in all blocks<br/>- No Stride<br/>- 2 Max pooling<br/>- Activation: RELU<br/>- LR: 0.05<br/>- Batch Size: 1024<br/>- Dropout: 0.15 | Dropout increased to 0.15 | Test accuracy > 99.4% achieved. Train accuracy is lesser than Validation accuracy. |

---

## 📈 7. Key Observations & Learnings

### 🎯 **Performance Achievements**
- **Final Accuracy**: 99.49% test accuracy achieved with only **19,348 parameters**
- **Target Met**: Successfully built a compact CNN under 20K parameters
- **Best Architecture**: 8-layer CNN with Global Average Pooling and strategic dropout

### 🔍 **Critical Insights**

#### **Architecture Design**
- **Kernel Size Impact**: Larger kernels (5x5) in early layers reduced accuracy due to excessive receptive field growth
- **Channel Progression**: Optimal pattern found: 16→32→16 channels with proper padding
- **Global Average Pooling**: Essential for parameter reduction while maintaining performance
- **Padding Strategy**: Consistent padding across all blocks crucial for feature map preservation

#### **Training Optimization**
- **Learning Rate**: Higher LR (0.05) with proper scheduling more effective than lower LR (0.001)
- **Batch Size**: Doubling from 512 to 1024 improved training stability
- **Scheduler**: StepLR with appropriate step size (5-10) significantly improved convergence
- **Validation Split**: Essential for proper model evaluation and overfitting detection

#### **Regularization Effects**
- **Batch Normalization**: Critical for training stability and convergence speed
- **Dropout Strategy**: Selective dropout (0.1-0.15) in final layers more effective than global dropout
- **Overfitting Control**: Train accuracy < Validation accuracy indicates good generalization

### 📊 **Parameter Efficiency Journey**
1. **Initial**: 12K params → 89.64% accuracy
2. **Expansion**: 36K params → 94.75% accuracy  
3. **Optimization**: 19K params → 99.49% accuracy

### 🚀 **Key Success Factors**
- **Progressive Architecture Refinement**: Systematic experimentation with each component
- **Data Augmentation**: Proper validation/test split with consistent preprocessing
- **Hyperparameter Tuning**: Learning rate and scheduler optimization
- **Regularization Balance**: Strategic use of BN and dropout without over-regularization


---

## 🧠 8. Important Learnings as Insights

### 💡 **Lessons Learned**
- **Parameter Count ≠ Performance**: Smart architecture design more important than parameter count
- **Validation is Key**: Proper train/validation split essential for reliable evaluation
- **Iterative Improvement**: Each experiment built upon previous learnings
- **Global Average Pooling**: Game-changer for parameter efficiency in CNNs

### 🏊 **Pooling Layer**

Pooling layers are fundamental in CNNs for efficient, robust visual feature learning. They help in:

1. **Transforming large feature maps into smaller, informative representations** by down sampling and **reduce computational burden**
2. **Encourage generalization** (Eliminating noisy features)
3. Build **translation-invariant feature hierarchies** essential for complex image recognition tasks.

#### What are pooling layers and their purpose?

Pooling layers are specialized layers in CNNs that **down sample the spatial dimensions** (width and height) of feature maps generated by convolutional layers. They summarize or aggregate local patches of the input feature map, producing smaller, condensed maps that retain the most important information.

**Key purposes of pooling layers:**

- **Dimensionality reduction:** Pooling layers reduce the width and height of feature maps, lowering computational cost and memory usage in the network. This parameter reduction reduces the chance of overfitting.
- **Translation invariance:** Pooling contributes to making the CNN robust to small shifts or distortions in the input image. The output after pooling remains stable even if the features relocate slightly in the input.
- **Feature abstraction:** By summarizing local regions, pooling layers help build hierarchical, abstract feature representations that are essential for recognizing complex patterns.
- **Noise reduction:** Pooling smoothens out irrelevant or noisy activations in feature maps, improving generalization.

### ⚖️ **What is Batch Normalization?**

Batch normalization (batch norm) is a technique used in neural networks like your CNN to stabilize and accelerate training by normalizing the activations (outputs) of a layer across a batch of data. It ensures that the inputs to each layer have a consistent mean and variance, which helps the network learn more effectively. Mathematically, it standardizes the outputs of a layer and then applies a learnable scale and shift.

In your CNN, batch norm is applied after each convolutional layer (e.g., conv1, conv2, etc.) to normalize the feature maps before passing them to the ReLU activation or the next layer.

### 🎯 **Why Do We Add Dropout?**

Dropout is added to neural networks for the following reasons:

1. **Prevent Overfitting:**
   - Overfitting occurs when a model learns to memorize the training data instead of generalizing to unseen data. Dropout mitigates this by introducing randomness, making the model less reliant on specific neurons and encouraging it to learn more robust, generalized features.

2. **Encourages Redundancy:**
   - By randomly dropping neurons, the model learns multiple independent pathways to make predictions. This mimics an ensemble of smaller networks, improving robustness and reducing dependency on any single feature or neuron.

3. **Improves Generalization:**
   - Dropout forces the network to work with a subset of features, simulating a form of noise in the data. This helps the model perform better on validation and test sets, as it learns to handle variations and incomplete information.

4. **Reduces Co-Adaptation:**
   - Without dropout, neurons can co-adapt, meaning they rely heavily on each other to produce correct outputs. Dropout breaks this co-dependency, encouraging each neuron to contribute independently to the model's predictions.

5. **Computationally Efficient Regularization:**
   - Compared to other regularization techniques (e.g., L1/L2 weight decay), dropout is simple to implement and computationally lightweight, as it only involves random masking and scaling.

#### When and Where to Use Dropout

- **Where:** Dropout is typically applied to **fully connected layers** (e.g., after flattening in a CNN or in dense layers of an MLP) or sometimes after **convolutional layers** in CNNs (though less common due to spatial correlations). In modern architectures, techniques like batch normalization often complement or replace dropout in conv layers.
- **When:** Use dropout when you observe overfitting (e.g., high training accuracy but low validation accuracy). It's most effective in deep networks with many parameters.
- **Typical Values:** Common dropout rates are p=0.2 to 0.5. Higher values (e.g., 0.5) are used in larger networks, while smaller values (e.g., 0.2) suffice for smaller models.


---

## ⚖️ **Data Normalization**

Normalization rescales the pixel intensity values to help the network train more effectively.

### �� Typical normalization for MNIST:
```python
transforms.Normalize((0.1307,), (0.3081,))
```

✅ **Benefits:**
- Speeds up convergence
- Prevents vanishing/exploding gradients
- Stabilizes learning across layers

---

## �� **Data Augmentation**

While MNIST is relatively clean, data augmentation helps improve generalization, especially for compact networks.

**Techniques Used:**
```python
transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

✅ **Advantages:**
- Reduces overfitting
- Improves robustness to real-world variations in handwriting
- Simulates rotation, translation, and distortion

---