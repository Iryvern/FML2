import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        # Adjust the input size to match 256x256 images (65536 pixels)
        self.encoder = nn.Sequential(
            nn.Linear(256 * 256, 128),  # Input size is now 65536
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256 * 256),  # Output size is now 65536
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input images from (batch_size, 1, 256, 256) to (batch_size, 65536)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        # Reshape the output back to (batch_size, 1, 256, 256)
        x = x.view(x.size(0), 1, 256, 256)
        return x

class GRUAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUAnomalyDetector, self).__init__()
        # GRU layer with specified number of layers and hidden size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layers to process the GRU output
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, output_size)  # Output layer for the final result
        )
        
    def forward(self, x):
        # Initialize hidden state for the GRU
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        
        # Forward propagate the GRU
        out, _ = self.gru(x, h0)
        
        # Take the last output of GRU for each sequence
        out = out[:, -1, :]
        
        # Pass through the fully connected layers
        out = self.fc_layers(out)
        return out