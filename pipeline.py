import csv
from pathlib import Path

from progress.bar import Bar

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from inaturalist import FISH_CLASSES, iNaturalistDataset


class Pipeline:
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        top_k: int,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.k = top_k

        if not torch.cuda.is_available():
            print("WARNING: Using CPU instead of GPU")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.path = Path("models") / f"model_{self.job_id}.pt"

        if self.path.exists():
            print("Loading model from", self.path)
            checkpoint = torch.load(self.path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.loss = checkpoint["loss"]
        else:
            if not Path("models").exists():
                Path("models").mkdir()
            print("Training model from scratch")
            self.epoch = 0

    def start_pipeline(self, job_id, download=False, classes=FISH_CLASSES):
        self.job_id = job_id

        self.data_setup(download, classes)
        self.load_model()
        self.train()
        self.evaluate()

    def data_setup(self, download, classes):
        self.all_transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
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
            transform=self.all_transforms,
        )
        test = iNaturalistDataset(
            root="./data",
            train=False,
            download=download,
            classes=classes,
            transform=self.all_transforms,
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
        with open(f"train_{self.job_id}.csv", mode="a", newline="") as file:
            writer = csv.writer(file)

            start_epoch = self.epoch
            with Bar("Training", max=self.num_epochs - start_epoch) as bar:
                for epoch in range(start_epoch, self.num_epochs):
                    self.epoch = epoch
                    for step, (images, labels) in enumerate(self.train_dataloader):
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        outputs = self.model(images)
                        self.loss = self.criterion(outputs, labels)

                        # Backward and optimize
                        self.optimizer.zero_grad()
                        self.loss.backward()
                        self.optimizer.step()

                    torch.save(
                        {
                            "epoch": self.epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": self.loss,
                        },
                        self.path,
                    )

                    loss = self.loss.item()
                    top1, topk = self.evaluate()

                    writer.writerow([epoch, loss, top1, topk])
                    bar.next()

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

            top1 = 100 * correct_top_1 / total
            topk = 100 * correct_top_k / total

            return top1, topk

    def predict(self):
        raise NotImplementedError()

