from imports import *
from models import SparseAutoencoder, YOLOv11  # Assuming Yolo11 model is defined
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# Train function
def train(net, trainloader, epochs: int, optimizer, model_type="autoencoder"):
    if model_type == "autoencoder":
        criterion = torch.nn.MSELoss()  # Loss for image reconstruction
    elif model_type == "yolo":
        criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropy for classification/detection task in YOLO
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    net.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in trainloader:
            images, labels = batch if model_type == "yolo" else (batch, None)
            images = images.to(DEVICE)
            optimizer.zero_grad()  # Reset gradients
            outputs = net(images)

            # Calculate the loss based on the model type
            if model_type == "autoencoder":
                loss = criterion(outputs, images)  # Autoencoder reconstruction loss
            elif model_type == "yolo":
                labels = labels.to(DEVICE)
                loss = criterion(outputs, labels)  # YOLO object detection loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()

        average_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: train loss {average_loss:.4f}")

# Test function
def test(net, testloader, model_type="autoencoder"):
    if model_type == "autoencoder":
        criterion = torch.nn.MSELoss()  # Loss for image reconstruction
    elif model_type == "yolo":
        criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropy for classification/detection task in YOLO
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    total_loss = 0.0
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch if model_type == "yolo" else (batch, None)
            images = images.to(DEVICE)
            outputs = net(images)

            # Calculate the loss based on the model type
            if model_type == "autoencoder":
                loss = criterion(outputs, images)  # Autoencoder reconstruction loss
            elif model_type == "yolo":
                labels = labels.to(DEVICE)
                loss = criterion(outputs, labels)  # YOLO object detection loss

            total_loss += loss.item()

    average_loss = total_loss / len(testloader)
    print(f"Test loss: {average_loss:.4f}")
    return average_loss
