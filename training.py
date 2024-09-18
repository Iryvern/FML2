from imports import *
from models import SparseAutoencoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Assuming trainloaders is provided or globally available

# Train function
def train(net, trainloader, epochs: int, optimizer):
    criterion = torch.nn.MSELoss()  # Define the loss function
    net.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in trainloader:
            images = batch.to(DEVICE)
            optimizer.zero_grad()  # Reset gradients
            outputs = net(images)
            loss = criterion(outputs, images)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()
        average_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: train loss {average_loss:.4f}")

# Test function
def test(net, testloader):
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    average_loss = total_loss / len(testloader)
    print(f"Test loss: {average_loss:.4f}")
    return average_loss
