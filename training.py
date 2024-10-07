from imports import *
from models import SparseAutoencoder, GRUAnomalyDetector  # Import the GRU model
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Assuming trainloaders is provided or globally available

# Train function
def train(net, trainloader, epochs: int, optimizer, data_type="image"):
    # Select appropriate loss function based on the data type
    criterion = torch.nn.MSELoss() if data_type == "image" else torch.nn.MSELoss()
    net.train()  # Set the model to training mode
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in trainloader:
            if data_type == "image":
                inputs = batch.to(DEVICE)  # For image data
                inputs = inputs.view(inputs.size(0), -1)  # Flatten images for the autoencoder
            elif data_type == "gps":
                inputs = batch.float().to(DEVICE)  # For GPS data, convert to float and move to the device
                
            optimizer.zero_grad()  # Reset gradients
            outputs = net(inputs)
            loss = criterion(outputs, inputs)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()
        
        average_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: train loss {average_loss:.4f}")

# Test function
def test(net, testloader, data_type="image"):
    # Select appropriate loss function based on the data type
    criterion = torch.nn.MSELoss() if data_type == "image" else torch.nn.MSELoss()
    total_loss = 0.0
    net.to(DEVICE)
    net.eval()
    
    with torch.no_grad():
        for batch in testloader:
            if data_type == "image":
                inputs = batch.to(DEVICE)  # For image data
                inputs = inputs.view(inputs.size(0), -1)  # Flatten images for the autoencoder
            elif data_type == "gps":
                inputs = batch.float().to(DEVICE)  # For GPS data, convert to float and move to the device
            
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    
    average_loss = total_loss / len(testloader)
    print(f"Test loss: {average_loss:.4f}")
    return average_loss
