import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn

from pipeline import Pipeline
from visual_transformer import VisualTransformer

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
