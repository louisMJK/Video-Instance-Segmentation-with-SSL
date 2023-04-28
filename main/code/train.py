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
import copy
import matplotlib.pyplot as plt

from data import ImagesToMaskDataset
from utils import criterion, init_distributed_mode, mkdir
from models import create_model
from transforms import SegmentationTrainTransform, SegmentationValTransform


parser = argparse.ArgumentParser(description='PyTorch Instance Segementation')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='fcn_resnet50', type=str)
group.add_argument('--backbone', default='resnet50', type=str)
group.add_argument('--backbone-dir', default='../../../ssl/output/backbone-resnet50-0.9167/resnet50_best.pth', type=str)
group.add_argument('--predictor-dir', default='', type=str)
group.add_argument('--fcn-dir', default='../../../segmentation/output/fcn_resnet50_jaccard_0.9204/', type=str)
group.add_argument('--freeze-backbone', action='store_true', default=False)
group.add_argument('--freeze-predictor', action='store_true', default=False)
group.add_argument('--freeze-fcn', action='store_true', default=False)
group.add_argument('--use-trained', action='store_true', default=False)
group.add_argument('--use-predictor', action='store_true', default=False)
group.add_argument('--use-fcn-head', action='store_true', default=False)

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='adam', type=str, metavar='OPTIMIZER') 
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=1e-5)
group.add_argument('--lr-base', type=float, default=1e-2, metavar='LR')
group.add_argument('--step-size', type=int, default=2)
group.add_argument('--lr-decay', type=float, default=0.9)

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
    print(f'\nTraining on {device}.')


    # Data augmentation
    transform_train = SegmentationTrainTransform()
    transform_val = SegmentationValTransform()

    # Dataset
    print("Loading dataset...")
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
        # collate_fn=utils.collate_fn,
    )
    loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.workers
    )
    dataloaders = {'train': loader_train, 'val': loader_val}


    # model
    model = create_model(args)
    model.to(device)
    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        if args.verbose:
            f.write(str(summary(model, input_size=(11, 3, 160, 240))))

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr_base, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model_without_ddp.parameters(), lr=args.lr_base, weight_decay=args.weight_decay)
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
    else:
        scheduler = None
    
    # loss function
    loss_fn = criterion


    # train
    print()
    print(f"Training {args.model} for {args.epochs} epochs ...")

    model_best, losses, ious = \
        train_model(model, loss_fn, optimizer, scheduler, dataloaders, dataset_sizes, train_sampler, device, args, model_without_ddp)
    
    print('-' * 30)
    
    # make plot and log
    plot_loss_and_acc(losses['train'], ious['train'], losses['val'], ious['val'], exp_dir + 'loss_acc.pdf')
    data = np.array([losses['train'], losses['val'], ious['train'], ious['val']])
    df_log = pd.DataFrame(data.T, columns=['loss_train', 'loss_val', 'jaccard_train', 'jaccard_val'])
    df_log.to_csv(exp_dir + str(max(ious['val'])) + "log.csv", index=False)



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
        model_without_ddp
):
    t_start = time.time()
    model.to(device)
    best_model_wts = copy.deepcopy(model_without_ddp.state_dict())
    best_jac = 0.0
    losses = {'train': [], 'val': []}
    jaccard_indices = {'train': [], 'val': []}
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49, average='micro')
    jaccard2 = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

    for epoch in range(args.epochs):
        t1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
            running_indices = 0.0
            running_indices2 = 0.0

            # Iterate over data
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.long())
                    # IoU
                    out = outputs['out']
                    masks_pred = out.cpu().detach().numpy().argmax(1)
                    jaccard_idx = jaccard(targets.cpu().detach(), torch.Tensor(masks_pred)).item()
                    jaccard_idx2 = jaccard2(targets.cpu().detach(), torch.Tensor(masks_pred)).item()


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if args.sched == 'poly':
                            scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_indices += jaccard_idx * inputs.size(0) * args.world_size
                running_indices2 += jaccard_idx2 * args.world_size

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_jac = running_indices / dataset_sizes[phase]
            epoch_jac2 = running_indices2 / len(dataloaders[phase])
            losses[phase].append(epoch_loss)
            jaccard_indices[phase].append(epoch_jac)

            if phase == 'train':
                print(f'Epoch [{epoch+1:4d}/{args.epochs:4d}], Train Loss: {epoch_loss:.3e}, Jaccard: {epoch_jac:.4f}, Jaccard2: {epoch_jac2:.4f}', end='')
            else:
                print(f'   Val Loss: {epoch_loss:.3e}, Jaccard: {epoch_jac:.4f}, Jaccard2: {epoch_jac2:.4f}   Time: {(time.time()-t1):.0f}s')

            # save best model
            if phase == 'val' and epoch_jac > best_jac:
                best_jac = epoch_jac
                best_model_wts = copy.deepcopy(model_without_ddp.state_dict())
                if dist.get_rank() == 0:
                    torch.save(best_model_wts, args.exp_dir + 'model_best.pth')
                    # torch.save(model_without_ddp.backbone.state_dict(), args.exp_dir + 'backbone.pth')
                    torch.save(model_without_ddp.predictor.state_dict(), args.exp_dir + 'predictor.pth')
                    torch.save(model_without_ddp.classifier.state_dict(), args.exp_dir + 'classifier.pth')
                    torch.save(model_without_ddp.aux_classifier.state_dict(), args.exp_dir + 'aux_classifier.pth')

    time_elapsed = time.time() - t_start
    print(f'Training completed in {time_elapsed // 60:.0f} min {time_elapsed % 60:.0f} s')
    print(f'Best Validation Jaccard: {best_jac:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, losses, jaccard_indices



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

