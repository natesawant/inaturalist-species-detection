from torch import nn
from torchvision import models
# from transformers import ViTForImageClassification


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

        # self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        self.model = models.vision_transformer.vit_h_14(weights=models.vision_transformer.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)

        # Freeze all layers except the final classification layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final classification layer
        num_ftrs = self.model.heads.head.in_features
        # self.model.heads.head = nn.Sequential(
        #     nn.Linear(num_ftrs, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(), 
        #     nn.Linear(256, num_classes)
        # )

        self.model.heads.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, img):
        return self.model.forward(img)
