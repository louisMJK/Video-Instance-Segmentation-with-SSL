import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, distributed
import torch.distributed as dist
from torchinfo import summary
import torchmetrics

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import argparse
import matplotlib.pyplot as plt

from utils import criterion, init_distributed_mode, mkdir, MetricLogger
from transforms import SegmentationTrainTransform, SegmentationValTransform
from data import ImagesToMaskDataset
from models.main_model import MainModel


parser = argparse.ArgumentParser(description='PyTorch Video Instance Segementation')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--predictor-dir', default='', type=str)
group.add_argument('--fcn-dir', default='', type=str)
group.add_argument('--main-model-dir', default=None, type=str)
group.add_argument('--freeze-predictor', action='store_true', default=False)
group.add_argument('--freeze-backbone', action='store_true', default=False)
group.add_argument('--freeze-fcn-head', action='store_true', default=False)

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='adam', type=str, metavar='OPTIMIZER') 
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=1e-6)
group.add_argument('--lr-base', type=float, default=1e-3, metavar='LR')
group.add_argument('--lr-pred', type=float, default=1e-5, metavar='LR')
group.add_argument('--lr-backbone', type=float, default=1e-2, metavar='LR')
group.add_argument('--lr-fcn', type=float, default=1e-3, metavar='LR')
group.add_argument('--lr-decay', type=float, default=0.9)
group.add_argument('--mode', type=str, default='triangular2')
group.add_argument('--epoch-size-up', type=int, default=10)

# Train
group = parser.add_argument_group('Training parameters')
group.add_argument('--epochs', type=int, default=40, metavar='N')
group.add_argument('-b', '--batch-size', type=int, default=16, metavar='N')
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
    print(f'\nTraining on {device}.\n')

    # Data augmentation
    transform_train = SegmentationTrainTransform()
    transform_val = SegmentationValTransform()

    # Dataset
    data_dir = args.data_dir
    dataset_train = ImagesToMaskDataset(os.path.join(data_dir, 'train'), transform_train)
    dataset_val = ImagesToMaskDataset(os.path.join(data_dir, 'val'), transform_val)
    dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}

    if args.distributed:
        train_sampler = distributed.DistributedSampler(dataset_train)
        val_sampler = distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = RandomSampler(dataset_train)
        val_sampler = SequentialSampler(dataset_val)

    # Dataloader
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
    )
    loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.workers
    )
    dataloaders = {'train': loader_train, 'val': loader_val}


    # model
    model = MainModel()
    
    # load trained weights
    if args.main_model_dir is not None:
        # model = torch.load(args.main_model_dir, map_location='cpu')
        model.load_state_dict(torch.load(args.main_model_dir, map_location='cpu'), strict=True)
    else:
        model.predictor.load_state_dict(torch.load(args.predictor_dir, map_location='cpu'), strict=True)
        model.fcn_resnet.load_state_dict(torch.load(args.fcn_dir, map_location='cpu'), strict=True)

    model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    # freeze
    if args.freeze_predictor:
        for p in model.predictor.parameters():
            p.requires_grad = False
        print('Predictor freezed.')
    if args.freeze_backbone:
        for p in model.fcn_resnet.backbone.parameters():
            p.requires_grad = False
        print('Backbone freezed.')
    if args.freeze_fcn_head:
        for p in model.fcn_resnet.classifier.parameters():
            p.requires_grad = False
        for p in model.fcn_resnet.aux_classifier.parameters():
            p.requires_grad = False
        print('FCN head freezed.')

    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        if args.verbose:
            f.write(str(summary(model, input_size=(3, 160, 240), batch_dim=0)))

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # params
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.predictor.parameters() if p.requires_grad], "lr": args.lr_pred},
        {"params": [p for p in model_without_ddp.fcn_resnet.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": [p for p in model_without_ddp.fcn_resnet.classifier.parameters() if p.requires_grad], "lr": args.lr_fcn},
    ]
    params = [p for p in model_without_ddp.fcn_resnet.aux_classifier.parameters() if p.requires_grad]
    params_to_optimize.append({"params": params, "lr": args.lr_fcn * 10})

    # optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr_base, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr_base, weight_decay=args.weight_decay)
    else:
        print('No optimizer selected !')

    # scheduler
    iters_per_epoch = len(loader_train)
    if args.sched == 'poly':
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=iters_per_epoch * (args.epochs), power=0.9)
    elif args.sched == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.sched == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    elif args.sched == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=args.lr_base / 10, 
            max_lr=args.lr_base,
            step_size_up=int(args.epoch_size_up * dataset_sizes['train'] / args.batch_size),
            mode=args.mode,
            cycle_momentum=(not (args.optim=='adam')),
        )
    else:
        scheduler = None
    
    # loss function
    loss_fn = criterion

    # train
    print(f"Training for {args.epochs} epochs ...\n")

    losses, ious = \
        train_model(model, loss_fn, optimizer, scheduler, dataloaders, dataset_sizes, train_sampler, device, args)
    
    print('-' * 30)
    
    # make plot and log
    plot_loss_and_acc(losses['train'], ious['train'], losses['val'], ious['val'], exp_dir + 'loss_acc.pdf')
    data = np.array([losses['train'], losses['val'], ious['train'], ious['val']])
    df_log = pd.DataFrame(data.T, columns=['loss_train', 'loss_val', 'jaccard_train', 'jaccard_val'])
    df_log.to_csv(exp_dir + "log.csv", index=False)



def train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        dataloaders, 
        dataset_sizes, 
        train_sampler,
        device,
        args,
):
    t_start = time.time()
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model_without_ddp = model.module

    best_jac = 0.0
    losses = {'train': [], 'val': []}
    jaccard_indices = {'train': [], 'val': []}
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

    for epoch in range(args.epochs):
        t1 = time.time()
        metric_logger = MetricLogger(delimiter="  ")

        if args.distributed:
            train_sampler.set_epoch(epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
            # running_indices = 0.0
            masks_pred = np.zeros((0, 160, 240))
            masks = np.zeros((0, 160, 240))

            # Iterate over data
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.long())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if args.sched == 'poly' or args.sched == 'cyclic':
                            scheduler.step()
                
                with torch.no_grad():
                    out = outputs['out']
                    masks_batch = out.cpu().detach().numpy().argmax(1)
                    masks_pred = np.concatenate([masks_pred, masks_batch], axis=0)
                    masks = np.concatenate([masks, targets.cpu().detach().numpy()], axis=0)
                    running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            jac = jaccard(torch.Tensor(masks_pred), torch.Tensor(masks))

            if phase == 'train':
                metric_logger.update(loss_train=epoch_loss, jac_train=jac, lr=optimizer.param_groups[0]["lr"])
            else:
                metric_logger.update(loss_val=epoch_loss, jac_val=jac, lr=optimizer.param_groups[0]["lr"])

            loss_avg = metric_logger.meters['loss_' + phase].avg * args.world_size
            jac_avg = metric_logger.meters['jac_' + phase].avg
            losses[phase].append(loss_avg)
            jaccard_indices[phase].append(jac_avg)

            if phase == 'train':
                print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}]   Train Loss: {loss_avg:.3e}, Jaccard: {jac_avg:.4f}', end='')
            else:
                print(f'   Val Loss: {loss_avg:.3e}, Jaccard: {jac_avg:.4f}   Time: {(time.time()-t1):.0f}s')

            # save best model
            if phase == 'val' and jac_avg > best_jac:
                best_jac = jac_avg
                if dist.get_rank() == 0:
                    torch.save(model_without_ddp.state_dict(), args.exp_dir + 'main_model_best.pth')

    time_elapsed = time.time() - t_start
    print(f'Training completed in {time_elapsed // 60:.0f} min {time_elapsed % 60:.0f} s')
    print(f'Best Validation Jaccard: {best_jac:.4f}')

    return losses, jaccard_indices



def plot_loss_and_acc(train_losses, train_accs, val_losses, val_accs, out_pth="model.pth"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].plot(train_losses)
    axs[0].plot(val_losses)
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(["Train loss", "Val loss"])
    axs[1].plot(train_accs)
    axs[1].plot(val_accs)
    axs[1].grid()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Jaccard Index')
    axs[1].legend(["Train Jaccard", "Val Jaccard"])
    axs[1].set_ylim([0, 1.0])
    
    fig.savefig(out_pth)



if __name__ == "__main__":
    main()

