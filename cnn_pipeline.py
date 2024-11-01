import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

from convolutional_neural_network import ConvolutionalNeuralNetwork
from pipeline import Pipeline

class CNNPipeline(Pipeline):
    def load_model(self):
        self.model = ConvolutionalNeuralNetwork(self.num_classes)
        # Set Loss function with criterion
        self.criterion = nn.CrossEntropyLoss()
        # Set optimizer with optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.005,
            momentum=0.9,
        )
        super().load_model()

    def train(self):
        self.model = self.model.to(self.device)
        return super().train()
    