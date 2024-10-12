from torch import nn


class VisualTransformer(nn.Module):
    def __init__(
        self,
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

    def forward(self, img):
        raise NotImplementedError()
