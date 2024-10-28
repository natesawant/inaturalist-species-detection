import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from convolutional_neural_network import ConvolutionalNeuralNetwork
from visual_transformer import VisualTransformer

from inaturalist import FISH_CLASSES, iNaturalistDataset


class Pipeline:
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        top_k: int = 5,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.k = top_k
        if not torch.cuda.is_available():
            print("WARNING: Using CPU instead of GPU")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start_pipeline(self, download=False, classes=FISH_CLASSES):
        self.data_setup(download, classes)
        self.train()
        self.evaluate()

    def data_setup(self, download, classes):
        all_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        train = iNaturalistDataset(
            root="./data",
            train=True,
            download=download,
            classes=classes,
            transform=all_transforms,
        )
        test = iNaturalistDataset(
            root="./data",
            train=False,
            download=download,
            classes=classes,
            transform=all_transforms,
        )

        self.train_dataloader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            test, batch_size=self.batch_size, shuffle=True
        )

        assert set(train.classes) == set(test.classes)
        self.num_classes = len(train.classes)

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
            correct_top_1 = 0
            correct_top_k = 0
            total = 0
            for images, labels in self.test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.topk(outputs.data, k=self.k, dim=1, largest=True)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_top_k += (
                    (predictions == labels.view(-1, 1)).any(dim=1).sum().item()
                )
                correct_top_1 += (predicted == labels).sum().item()

            print(
                "Top-1 Accuracy of the network on the {} test images: {} %".format(
                    len(self.test_dataloader), 100 * correct_top_1 / total
                )
            )
            print(
                "Top-{} Accuracy of the network on the {} test images: {} %".format(
                    self.k, len(self.test_dataloader), 100 * correct_top_k / total
                )
            )

    def predict(self):
        raise NotImplementedError()


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


def parse_args():
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Boolean argument for "download"
    parser.add_argument(
        "--download", action="store_true", help="Flag to initiate download"
    )

    # String list argument for "classes"
    parser.add_argument("--classes", type=str, nargs="+", help="List of class names")

    # Parse the arguments
    args = parser.parse_args()

    return args.download, args.classes


if __name__ == "__main__":
    download, classes = parse_args()
    image_size = 32
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20

    pipeline = CNNPipeline(
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    pipeline.start_pipeline(download=download, classes=classes)
