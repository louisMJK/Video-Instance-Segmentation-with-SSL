from torch import optim


def create_optimizer(model, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr_base, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    elif args.optim == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr_base, 
            weight_decay=args.weight_decay
        )
    
    elif args.optim == 'lars':
        from torchlars import LARS
        base_optimizer = optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr_base, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        optimizer = LARS(base_optimizer)

    return optimizer

