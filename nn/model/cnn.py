import torch
from torch import nn
from torch.nn import functional as F


class CNN5(nn.Module):

    def __init__(self, input_dim=(3, 32, 32), out_classes=10):
        # CIFAR images are 32x32 pixels with 3 color channels
        super().__init__()
        self.input_dim = input_dim  # Flatten the 32x32x3 image into a single vector
        self.out_classes = out_classes

        self.conv1 = nn.Conv2d(
            in_channels=self.input_dim[0],
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)

        conv_out_size = self.calculate_conv_output_size(image_height=input_dim[1], image_width=input_dim[2])
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, self.out_classes)
        self.dropout = nn.Dropout(p=0.1)

    def calculate_conv_output_size(self, image_height: int, image_width: int):
        # This function calculates the output size after two convolutions and pooling
        height, width = image_height, image_width

        # First convolution reduces size
        height = (height - 5) // 1 + 1
        width = (width - 5) // 1 + 1

        # First max-pooling halves the size
        height = height // 2
        width = width // 2

        # Second convolution reduces size
        height = (height - 5) // 1 + 1
        width = (width - 5) // 1 + 1

        # Second max-pooling halves the size again
        height = height // 2
        width = width // 2

        # The final output size after conv layers and pooling
        return 64 * height * width

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
