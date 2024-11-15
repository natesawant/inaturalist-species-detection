from torch import nn
from torchvision import models


class VisualTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        ch=3,
        img_size=144,
        patch_size=4,
        emb_dim=32,
        n_layers=6,
        out_dim=37,
        dropout=0.1,
        heads=2,
    ):
        super(VisualTransformer, self).__init__()

        self.model = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")

        # Freeze all layers except the final classification layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final classification layer
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, img):
        return self.model.forward(img)
