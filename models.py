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

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample by a factor of 2

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample by a factor of 2

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample by a factor of 2

            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Down-sample by a factor of 2
        )

        # Calculate the flattened size based on the downsampling structure
        self.flattened_size = 256 * (256 // (2**4)) * (256 // (2**4))  # 256 * 16 * 16 = 65536
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # Apply the feature extractor (convolutions)
        x = x.view(x.size(0), -1)  # Flatten for the classifier
        x = self.classifier(x)
        return x