from torch import nn
from torchvision import models
from .simvp_model import SimVP_Model

class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = SimVP_Model()
        self.fcn_resnet = models.segmentation.fcn_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=49,
            aux_loss=True,
        )

    def forward(self, x):
        x = self.predictor(x)
        x = x[:, -1, :, :, :]
        x = self.fcn_resnet(x)
        return x

