import torch
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

    model.backbone.load_state_dict(adapted_dict, strict=True)
    print(f'Backbone loaded from {args.backbone_dir}.')

    if args.freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model

