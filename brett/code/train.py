import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

import os
import time
import numpy as np
from datetime import datetime
import yaml
import argparse
import csv

from torchvision.models import resnet18, resnet50, vit_b_16
from optimizer import create_optimizer
from optimizer import create_optimizer
from utils import UnlabeledDataset, BYOL
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.data import LightlyDataset


DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--backbone', default='resnet18', type=str, metavar='BACKBONE')
group.add_argument('--img-size', type=int, default=224)
group.add_argument('--img-val-resize', type=int, default=256)

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='sgd', type=str, metavar='OPTIMIZER')
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--lr-base', type=float, default=0.06, metavar='LR')
group.add_argument('--step-size', type=int, default=2)
group.add_argument('--lr-decay', type=float, default=0.9)

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--epochs', type=int, default=10, metavar='N')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N')
group.add_argument('--workers', type=int, default=4, metavar='N')
group.add_argument('--inference', action='store_true', default=True)
group.add_argument('--data-dir', default='../../dataset/', type=str)
group.add_argument('--out-dir', default='../../output/', type=str)
group.add_argument('--verbose', action='store_true', default=False)



def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text



def main():
    print('-' * 30)

    args, args_text = _parse_args()

    # logging
    out_dir = args.out_dir
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S")])
    exp_dir = out_dir + exp_name + "/"
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Training on 1 device ({device}).')

    # model
    if args.backbone == 'resnet50':
        backbone = resnet50()
    elif args.backbone == 'vit_b_16':
        backbone = vit_b_16()
    else:
        backbone = resnet18()

    model = BYOL(nn.Sequential(*list(backbone.children())[:-1]))
    model.to(device)


    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(str(summary(model, input_size=(3, args.img_size, args.img_size), batch_dim=0)))


    # optimizer
    optimizer = create_optimizer(model, args)

    # loss function
    loss_fn = NegativeCosineSimilarity()

    # scheduler
    if args.sched == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.sched == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, verbose=args.verbose)
    elif args.sched == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, 
                                              gamma=args.lr_decay, verbose=args.verbose)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)



    # Dataset
    print("Loading dataset...")
    data_dir = args.data_dir
    trans = SimCLRTransform(input_size=130)
    unlabeleddataset = UnlabeledDataset(root=os.path.join(data_dir, 'unlabeled'))
    dataset = LightlyDataset.from_torch_dataset(unlabeleddataset, transform=trans)
    dataset_size = len(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=MultiViewCollate(),
        shuffle=True, 
        drop_last=True, 
        num_workers=args.workers)


    #loss function
    criterion = NegativeCosineSimilarity()

    # train
    print("Starting Training")
    print(f"Training {args.backbone} for {args.epochs} epochs ...")
    model.to(device)
    model.train()  
    losses = []
    best_loss = np.inf

    #main loop
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0
        momentum_val = cosine_schedule(epoch, args.epochs, 0.996, 1)
        for (x0, x1), _, _ in dataloader:
            update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
            update_momentum(
                model.projection_head, model.projection_head_momentum, m=momentum_val
            )
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}, time:{time.time()-start_time:.2f}")
        scheduler.step()
        
        #save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, exp_dir + "model_best.pth")
            torch.save(backbone, exp_dir+"backbone_best.pth")
            print("Best model saved with loss: ", best_loss)
            print('-' * 30)
        losses.append(avg_loss)
        
        
    
    print("Training finished")
    print('-' * 30)

    #write loss to csv
    with open('loss.csv', 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow('loss')
        csvwriter.writerow(losses)


if __name__ == "__main__":
    main()

