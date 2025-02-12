import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()

        # Convolutional Layer 1: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Convolutional Layer 2: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # Pooling Layer (common to reuse a single pool op)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers:
        # After two convolution + pooling layers on 28x28, the feature map size is 7x7
        # 64 (channels) * 7 (height) * 7 (width) = 3136 features
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Convolutional block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28x28 → 14x14

        # Convolutional block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14x14 → 7x7

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 64*7*7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
