import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256 * 256, 128),
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
            nn.Linear(128, 256 * 256),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 256, 256)  # Reshape
        return x

class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(MobileNetV3, self).__init__()
        # Load MobileNetV3 with pretrained weights
        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)