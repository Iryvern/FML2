from ultralytics import YOLO
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

class YOLOv11(nn.Module):
    def __init__(self, model_path: str):
        super(YOLOv11, self).__init__()
        # Load YOLOv11 model with the pretrained weights
        self.model = YOLO(model_path)

    def forward(self, images):
        # Run inference
        results = self.model(images)
        return results
    
