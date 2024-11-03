import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from pipeline import Pipeline

class EfficientNetPipeline(Pipeline):
    def load_model(self):
        pretrained = True
        fine_tune = True

        if pretrained:
            print('[INFO]: Loading pre-trained weights')
        else:
            print('[INFO]: Not loading pre-trained weights')
        self.model = models.efficientnet_b0(pretrained=pretrained)
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False
        # Change the final classification head.
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Loss function.
        self.criterion = nn.CrossEntropyLoss()
        super().load_model()

    def train(self):
        self.model = self.model.to(self.device)
        return super().train()
    