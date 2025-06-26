import torch
import torch.nn.functional as F
from torch import nn


class CifarMlp(nn.Module):
    def __init__(self, input_dim: tuple = (3, 32, 32), out_classes: int = 10):
        # CIFAR images are 32x32 pixels with 3 color channels
        super().__init__()
        self.input_dim = input_dim  # Flatten the 32x32x3 image into a single vector
        self.out_classes = out_classes

        # Define the MLP architecture
        self.fc1 = nn.Linear(self.input_dim[0] * self.input_dim[1] * self.input_dim[2], 512)  # First dense layer
        self.fc2 = nn.Linear(512, 256)  # Second dense layer
        self.fc3 = nn.Linear(256, 128)  # Third dense layer
        self.fc4 = nn.Linear(128, 64)  # Fourth dense layer
        self.fc5 = nn.Linear(64, self.out_classes)  # Output layer

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
