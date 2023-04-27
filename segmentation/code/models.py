import torch
from torch import nn
from torchvision import models
from collections import OrderedDict


def create_model(args):
    model = models.segmentation.fcn_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=49,
        aux_loss=True,
    )

    state_dict = torch.load(args.backbone_dir, map_location='cpu')

    model_backbone_keys = list(model.backbone.state_dict().keys())
    backbone_keys = list(state_dict.keys())
    key_dict = {backbone_keys[i]: model_backbone_keys[i] for i in range(len(backbone_keys))}
    adapted_dict = OrderedDict({key_dict[k] : v for k, v in state_dict.items()})

    backbone = model.backbone
    backbone.load_state_dict(adapted_dict, strict=True)
    print(f'Backbone loaded from {args.backbone_dir}.')

    model = FCN(backbone)

    if args.freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model


class FCN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        fcn_resnet = models.segmentation.fcn_resnet50(num_classes=49, aux_loss=True)
        self.backbone = backbone
        self.fcn_head = nn.Sequential(
            fcn_resnet.classifier,
            nn.ConvTranspose2d(49, 49, kernel_size=16, padding=4, stride=8)
        )
        self.fcn_head_aux = nn.Sequential(
            fcn_resnet.aux_classifier,
            nn.ConvTranspose2d(49, 49, kernel_size=16, padding=4, stride=8)
        )

    def forward(self, x):
        x = self.backbone(x)         
        x1 = x['out']  # (B, 2048, 20, 30)
        x2 = x['aux']  # (B, 1024, 20, 30)
        y = {}
        y['out'] = self.fcn_head(x1)
        y['aux'] = self.fcn_head_aux(x2)
        return y

