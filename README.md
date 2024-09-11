# FashionMNIST Image Classification using CNNs

## Overview
This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify images from the **FashionMNIST** dataset, which consists of 28x28 grayscale images of clothing items. This project was inspired by a Kaggle notebook and served as a learning tool to understand the basics of **PyTorch** and **CNNs**.

The model achieved **~93% accuracy** on the validation set, and while the project is not entirely original, it was crucial in helping me grasp key concepts in deep learning and PyTorch.

## Project Goals
- Study and implement a **CNN architecture** using PyTorch.
- Understand how to manage and process image data with PyTorch’s `DataLoader` and `torchvision`.
- Explore **optimization techniques** (Adam optimizer) and **loss functions** (Cross-Entropy Loss).
- Learn to evaluate model performance using **accuracy metrics** and a **confusion matrix**.

## Model Architecture
The model follows the classic **LeNet-5 architecture**:
- **Conv2d**: Input 1 channel, Output 6 channels, Kernel size 5x5.
- **ReLU** activation.
- **AvgPool2d**: Kernel size 2x2, stride 2.
- **Conv2d**: Input 6 channels, Output 16 channels, Kernel size 5x5.
- **ReLU** activation.
- **AvgPool2d**: Kernel size 2x2, stride 2.
- **Fully connected layers**: Flatten input, 400 → 120 → 84 → 10.
- **ReLU** activation between fully connected layers.

## Training the Model
The model is trained using the **Adam optimizer** and **Cross-Entropy Loss**. The training process includes:

- **Epochs**: Default is 40, but can be adjusted.
- **Batch size**: Set to 64 for efficient GPU utilization.
- **Validation**: After each epoch, the validation accuracy is computed and stored.

You can train the model by calling the following function:
```python
best_model = train(numb_epoch=40, device=device)
