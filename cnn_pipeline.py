import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

from convolutional_neural_network import ConvolutionalNeuralNetwork
from pipeline import Pipeline

class CNNPipeline(Pipeline):

    def __init__(self, image_size=32, **kwargs):
        self.all_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        super().__init__(**kwargs)

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
    