# Homework Assignment #1: The one with the CNNs

## Overview

This assignment focuses on developing skills to work with convolutional neural networks (CNNs) through three progressive tasks:

1. Implementing the forward pass of a convolutional filter manually
2. Developing a custom CNN for image classification
3. Fine-tuning an existing CNN architecture for a similar task

## Task 1: Manual Implementation of Convolutional Forward Pass [2 points]

Implement the forward pass of a convolutional layer to understand the underlying mechanics.

### Requirements:
- Complete the implementation in the provided `conv.py` file
- Implement forward pass for two classes:
  - `MyConvStub`: A general convolution operation with parameters for:
    - Kernel size
    - Input/output channels
    - Bias term usage
    - Stride
    - Dilation
    - Grouping
  - `MyFilterStub`: A channelwise blur filter application

### Objectives:
1. Pass all convolution unit tests in `test_conv.py`
2. Correctly apply blur filters to input volumes

## Task 2: Custom CNN for Image Classification [4 points]

Design, implement, train and evaluate a simple CNN for image classification using the Imagenette dataset.

### Dataset:
- **Imagenette**: 160×160 px version
- 10 easily classified classes from ImageNet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute)

### Requirements:
- Design a neural network with at most 5 layers deep (excluding pooling, normalization, and activation functions)
- Justify your choice of activation functions and pooling layers
- Follow PyTorch's "Build Model" and "Classifying CIFAR10" tutorials

### Experiments:
Train your model with multiple configurations:
1. With and without batch normalization after convolution layers
2. With and without dropout regularization in final linear layers
3. With and without data augmentation methods
4. With all three techniques combined (batch norm, dropout, augmentation)

### Required Results:
- Training/Test Loss curves and Accuracy curves for each experiment
- Comparative plots showing each technique's impact
- Confusion matrix for the 10 classes
- Achieve at least 60% accuracy on test data
- Train for a minimum of 20 epochs

## Task 3: Transfer Learning with ResNet-18 [4 points]

Fine-tune a pre-trained convolutional neural network for the same image classification task.

### Requirements:
- Use ResNet-18 (pre-trained on ImageNet) as the backbone
- Follow PyTorch's transfer learning tutorial to use ResNet-18 as a feature extractor
- Experiment with unfreezing BatchNormalization layers to adapt statistics to the new dataset

### Required Results:
- Training/Test Loss curves
- Accuracy curves
- Confusion matrix
- Comparative analysis with your custom model from Task 2
- Analysis of BatchNorm layer unfreezing impact on performance

## Submission Guidelines

Submit the following:
- Completed implementation files
- Notebook or script files for model training and evaluation
- Report containing:
  - Model architecture descriptions and justifications
  - All required plots and visualizations
  - Analysis of results
  - Comparative discussions between experiments
  - Conclusions about employed techniques

## Resources

- PyTorch documentation
- "Build Model" and "Classifying CIFAR10" tutorials from PyTorch
- Transfer learning tutorial from PyTorch
