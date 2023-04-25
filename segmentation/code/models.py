import torch
import torchvision
from torchvision import models


def create_model(backbone, args):
    
    # model = models.segmentation.fcn_resnet50(
    #     weights=None,
    #     weights_backbone=backbone.state_dict(),
    #     num_classes=49,
    #     aux_loss=True,
    # )

    model = models.segmentation.fcn_resnet50(
        weights=None,
        weights_backbone=models.ResNet50_Weights.IMAGENET1K_V2,
        num_classes=49,
        aux_loss=True,
    )

    if args.freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model

