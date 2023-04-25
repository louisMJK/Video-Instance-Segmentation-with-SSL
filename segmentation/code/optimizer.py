from torch import optim


def create_optimizer(model, args):
    # fcn_resnet50
    params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]
    params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    params_to_optimize.append({"params": params, "lr": args.lr_base * 10})
    

    if args.optim == 'sgd':
        optimizer = optim.SGD(
            params_to_optimize,
            lr=args.lr_base, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    elif args.optim == 'adam':
        optimizer = optim.Adam(
            params_to_optimize,
            lr=args.lr_base, 
            weight_decay=args.weight_decay
        )

    return optimizer

