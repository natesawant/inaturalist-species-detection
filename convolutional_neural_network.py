from torch import nn


import torch
import torch.nn as nn

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes, image_size):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        # Define the convolutional layers
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the input size for the first fully connected layer
        self.fc_input_size = self._get_fc_input_size(image_size)

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _get_fc_input_size(self, image_size):
        # Create a dummy tensor to pass through the network to determine the output size
        x = torch.zeros(1, 3, image_size, image_size)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.max_pool1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)
        return x.numel()  # Get the number of elements in the output tensor

    def forward(self, img):
        out = self.conv_layer1(img)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.view(out.size(0), -1)  # Use view instead of reshape for clarity

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
