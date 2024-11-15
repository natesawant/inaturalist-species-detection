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

    def load_model(self):
        self.model = VisualTransformer(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        super().load_model()

    def train(self):
        self.model = self.model.to(self.device)
        return super().train()
