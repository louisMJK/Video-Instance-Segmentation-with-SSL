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

    return optimizer

