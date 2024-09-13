# FashionMNIST Image Classification using PyTorch

## Overview
This project implements a neural network using **PyTorch** to classify images from the **FashionMNIST** dataset. The dataset consists of 28x28 grayscale images of 10 different clothing items (like T-shirts, shoes, and bags). This project demonstrates how to use PyTorch's **DataLoader**, build a neural network with **nn.Sequential**, and train a model using **gradient descent**.

This implementation serves as a practical guide to understanding the basics of **PyTorch** for image classification tasks.

## Project Goals
- Study and implement a neural network using **PyTorch** for image classification.
- Understand how to manage and process image data with PyTorch's `DataLoader` and `torchvision`.
- Explore **optimization techniques** using **SGD (Stochastic Gradient Descent)**.
- Track training progress by monitoring the **loss** function to ensure the model is learning effectively.

## Dataset
The **FashionMNIST** dataset is used in this project. It consists of:
- **Training Set**: 60,000 images of 28x28 grayscale clothing items.
- **Test Set**: 10,000 images for evaluation.
- **Classes**: 10 fashion categories (T-shirt, trouser, pullover, dress, etc.).

The dataset is loaded using `torchvision.datasets.FashionMNIST`, and images are converted into tensors using `ToTensor()`.

## Model Architecture
The model is a simple fully connected neural network with multiple layers:
1. **Input Layer**: The input images are flattened from 28x28 pixels to a 784-dimensional vector.
2. **Hidden Layers**:
   - Two fully connected layers, each followed by **ReLU** activation functions.
   - These layers progressively transform the 784 input features into 512 hidden features and maintain this dimensionality.
3. **Output Layer**: A final fully connected layer maps the 512 hidden features to 10 output classes.

### Layer Details:
```python
self.flatten = nn.Flatten()
self.fc1 = nn.Linear(in_features=28*28, out_features=512)
self.fc2 = nn.Linear(in_features=512, out_features=512)
self.fc3 = nn.Linear(in_features=512, out_features=10)
self.relu = nn.ReLU()
```

## Training the Model
The training loop processes batches of data, computes the loss, and adjusts the model parameters via **backpropagation** and **gradient descent**.

### Steps:
1. **Forward Pass**: The input images pass through the model to generate predictions.
2. **Loss Calculation**: The model's predictions are compared to the true labels using a loss function (Cross-Entropy Loss).
3. **Backpropagation**: The gradients are calculated and stored.
4. **Optimizer Step**: The optimizer (SGD) updates the weights based on the gradients.
5. **Gradients Reset**: `optimizer.zero_grad()` is called to reset the gradients before the next iteration.

### Code Snippet:
```python
def training(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_func(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(x)
            print(f"Loss: {loss:.4f} [{current}/{size}]")
```
## Optimizer and Loss Function
- **Optimizer**: The **SGD (Stochastic Gradient Descent)** optimizer is used to update the model's weights based on the gradients calculated during backpropagation. The learning rate (`lr`) controls how much the weights are adjusted during each update step.
  
  Example code:
  ```python
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  ```
- **Loss Function**: The **Cross-Entropy Loss** function is used to calculate the error between the predicted output and the actual labels. This loss function is commonly used for multi-class classification tasks, like FashionMNIST.
  
  Example code:
  ```python
  loss_func = nn.CrossEntropyLoss()
  ```
## Evaluation
Before testing the model, it's important to switch the model to **evaluation mode** by calling:
```python
  model.eval()
```

This ensures that certain layers, such as **dropout** and **batch normalization**, behave correctly during testing. In evaluation mode, dropout is disabled, and batch normalization uses running statistics instead of batch statistics.

Additionally, during inference, you can use the following code block to prevent PyTorch from calculating gradients (as it's unnecessary and saves memory during evaluation):

```python
  with torch.no_grad():
    predictions = model(x)
```

