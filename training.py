from imports import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# Train function
def train(net, trainloader, epochs: int, optimizer, model_type):
    if model_type == "Image Anomaly Detection":
        criterion = torch.nn.MSELoss()  # Loss for image reconstruction
    elif model_type == "Image Classification":
        criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropy for classification task
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    net.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0.00
        for batch in trainloader:
            # Adjust input and target based on model type
            images, labels = batch if model_type == "Image Classification" else (batch, None)
            images = images.to(DEVICE)
            optimizer.zero_grad()  # Reset gradients
            outputs = net(images)

            # Calculate the loss based on the model type
            if model_type == "Image Anomaly Detection":
                loss = criterion(outputs, images)  # Autoencoder reconstruction loss
            elif model_type == "Image Classification":
                labels = labels.to(DEVICE)
                loss = criterion(outputs, labels)  # SimpleCNN classification loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()

        average_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: train loss {average_loss:.4f}")

# Test function
def test(net, testloader, model_type):
    if model_type == "Image Anomaly Detection":
        criterion = torch.nn.MSELoss()  # Loss for image reconstruction
    elif model_type == "Image Classification":
        criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropy for classification task
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    total_loss = 0.00
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            # Adjust input and target based on model type
            images, labels = batch if model_type == "Image Classification" else (batch, None)
            images = images.to(DEVICE)
            outputs = net(images)

            # Calculate the loss based on the model type
            if model_type == "Image Anomaly Detection":
                loss = criterion(outputs, images)  # Autoencoder reconstruction loss
            elif model_type == "Image Classification":
                labels = labels.to(DEVICE)
                loss = criterion(outputs, labels)  # SimpleCNN classification loss

            total_loss += loss.item()

    average_loss = total_loss / len(testloader)
    print(f"Test loss: {average_loss:.4f}")
    return average_loss