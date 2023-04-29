from torch import nn
from torchvision import models
from simvp_model import SimVP_Model


class LargeModel(nn.Module):
    """Large Model for the End to End Task"""

    def __init__(self):
        super(LargeModel, self).__init__()
        self.predictor = SimVP_Model()
        self.fcn_resnet = models.segmentation.fcn_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=49,
            aux_loss=True,
        )


    def forward(self, x, **kwargs):
        x = self.predictor(x)
        x = self.fcn_resnet(x)
        return x

