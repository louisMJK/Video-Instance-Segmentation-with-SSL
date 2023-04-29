import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, distributed
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as dist
from torchinfo import summary

import os
import time
import numpy as np
from datetime import datetime
import yaml
import argparse
import pandas as pd

from transforms import Transform
from optimizer import create_optimizer
from models import create_model
from utils import UnlabeledDataset, mkdir, init_distributed_mode
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightly.data import LightlyDataset



parser = argparse.ArgumentParser(description='PyTorch Self-Supervised Learning')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--backbone', default='resnet50', type=str, metavar='BACKBONE')
group.add_argument('--use-trained', action='store_true', default=False)
group.add_argument('--model-dir', default='', type=str)

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='sgd', type=str, metavar='OPTIMIZER') 
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=1e-6)
group.add_argument('--lr-base', type=float, default=1e-2, metavar='LR')
 
# Training
group = parser.add_argument_group('Training parameters')
group.add_argument('--epochs', type=int, default=40, metavar='N')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N')
group.add_argument('--workers', type=int, default=4, metavar='N')
group.add_argument('--data-dir', default='../../../dataset/', type=str)
group.add_argument('--out-dir', default='../../output/', type=str)
group.add_argument('--verbose', action='store_true', default=False)
group.add_argument("--dist-url", default="env://", type=str)
group.add_argument("--world-size", default=1, type=int)



def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text



def main():
    args, args_text = _parse_args()
    out_dir = args.out_dir
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S")])
    exp_dir = out_dir + exp_name + "/"
    args.exp_dir = exp_dir
    mkdir(exp_dir)
    
    # distributed 
    init_distributed_mode(args)
    print(args)

    # logging
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'\nTraining on {device}.')


    # model
    model = create_model(args)
    model.to(device)

    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        if args.verbose:
            f.write(str(summary(model, input_size=(3, 160, 240), batch_dim=0)))
    
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # optimizer
    optimizer = create_optimizer(model_without_ddp, args)

    # scheduler
    if args.sched == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.sched == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, verbose=args.verbose)
    elif args.sched == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, 
                                              gamma=args.lr_decay, verbose=args.verbose)
    else:
        scheduler = None


    # Dataset
    print("Loading dataset...")
    data_dir = args.data_dir
    trans = Transform(input_size=(160,240), min_scale=0.8)
    dataset = UnlabeledDataset(root=os.path.join(data_dir, 'unlabeled'))
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=trans)

    if args.distributed:
        train_sampler = distributed.DistributedSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        collate_fn=MultiViewCollate(),
        drop_last=True, 
        num_workers=args.workers
    )


    # loss function
    criterion = NegativeCosineSimilarity()

    # train
    print(f"Training {args.backbone} for {args.epochs} epochs ...\n")
    model.to(device)
    model.train()  
    losses = []
    best_loss = np.inf

    # resume from trained model
    if args.use_trained:
        model_without_ddp.load_state_dict(
            torch.load(args.model_dir, map_location='cpu'), 
            strict=True
        )   
        print(f'BYOL with {args.backbone} backbone loaded from {args.model_dir}.\n')       


    # main loop
    for epoch in range(args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        momentum_val = cosine_schedule(epoch, args.epochs, 0.996, 1)

        for (x0, x1), _, _ in dataloader:
            update_momentum(model_without_ddp.backbone, model_without_ddp.backbone_momentum, m=momentum_val)
            update_momentum(model_without_ddp.projection_head, model_without_ddp.projection_head_momentum, m=momentum_val)
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model_without_ddp.forward_momentum(x0)
            p1 = model(x1)
            z1 = model_without_ddp.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}]   Loss: {avg_loss:.3e},  Time: {time.time()-start_time:.0f}s')

        #save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if dist.get_rank() == 0:
                torch.save(model_without_ddp.state_dict(), exp_dir + 'byol_best.pth')
                torch.save(model_without_ddp.backbone.state_dict(), exp_dir + str(args.backbone) + '_best.pth')

        losses.append(avg_loss)
        
    print("Training completed.")
    print('-' * 30)

    if dist.get_rank() == 0:
        torch.save(model_without_ddp.state_dict(), exp_dir + 'byol_' + str(best_loss) + '.pth')
    print(f'Best models saved, loss: {best_loss:.4e}\n')

    # log loss
    data = np.array(losses)
    df_log = pd.DataFrame(data.T, columns=['loss'])
    df_log.to_csv(exp_dir + "log.csv", index=False)



if __name__ == "__main__":
    main()

