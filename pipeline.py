from pathlib import Path

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from convolutional_neural_network import ConvolutionalNeuralNetwork
from visual_transformer import VisualTransformer

import inaturalist


class Pipeline:
    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        learning_rate: float,
        num_epochs: int,
    ):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start_pipeline(self):
        self.data_setup()
        self.train()
        self.evaluate()

    def data_setup(self):
        all_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        train = inaturalist.iNaturalistDataset(
            root="./data",
            train=True,
            download=False,
            classes=["Elasmobranchii"],  # inaturalist.FISH_CLASSES,
            transform=all_transforms,
        )
        test = inaturalist.iNaturalistDataset(
            root="./data",
            train=False,
            download=False,
            classes=["Elasmobranchii"],  # inaturalist.FISH_CLASSES,
            transform=all_transforms,
        )

        self.train_dataloader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            test, batch_size=self.batch_size, shuffle=True
        )

    def train(self):
        for epoch in range(self.num_epochs):
            for step, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, self.num_epochs, loss.item()
                )
            )

    def evaluate(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                "Accuracy of the network on the {} train images: {} %".format(
                    len(self.train_dataloader), 100 * correct / total
                )
            )

    def predict(self):
        raise NotImplementedError()


class CNNPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.all_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        super().__init__(**kwargs)

    def train(self):
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
        self.model = self.model.to(self.device)
        return super().train()


class ViTPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.all_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),  # Maybe should be 144, 144
                transforms.ToTensor(),
            ]
        )
        super().__init__(**kwargs)

    def train(self):
        self.model = VisualTransformer().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return super().train()


if __name__ == "__main__":
    pipeline = CNNPipeline(
        batch_size=64, num_classes=16, learning_rate=0.001, num_epochs=20
    )
    pipeline.start_pipeline()
