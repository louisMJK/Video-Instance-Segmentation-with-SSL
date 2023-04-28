import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader, distributed, RandomSampler, SequentialSampler
import torch.distributed as dist
from torchmetrics.functional import structural_similarity_index_measure

import os
import time
import numpy as np
from datetime import datetime
import yaml
import argparse
import copy
import pandas as pd
import matplotlib.pyplot as plt
import random

from utils import init_distributed_mode, mkdir
from simvp_model import SimVP_Model
from optimizer import create_optimizer
from data import UnlabledtrainpredSim
from torchinfo import summary


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--width', default='240', type=int, metavar='WIDTH')
group.add_argument('--height', default='160', type=int, metavar='HEIGHT')

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='adam', type=str, metavar='OPTIMIZER')
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--lr-base', type=float, default=0.001, metavar='LR')
group.add_argument('--step-size', type=int, default=2)
group.add_argument('--lr-decay', type=float, default=0.9)

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--epochs', type=int, default=2, metavar='N')
group.add_argument('-b', '--batch-size', type=int, default=2, metavar='N')
group.add_argument('--workers', type=int, default=4, metavar='N')
group.add_argument('--inference', action='store_true', default=True)
group.add_argument('--data-dir', default='../../small_dataset/', type=str)
group.add_argument('--out-dir', default='../../output/', type=str)
group.add_argument('--verbose', action='store_true', default=False)
group.add_argument('--best-model', default=None, type=str)
group.add_argument('--sample-interval', type=int, default=1)
group.add_argument("--dist-url", default="env://", type=str)
group.add_argument("--world-size", default=1, type=int)

#model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--hid-S', default=128, type=int, metavar='hid_S')
group.add_argument('--hid-T', default=512, type=int, metavar='hid_T')
group.add_argument('--N-T', default=8, type=int, metavar='N_T')
group.add_argument('--N-S', default=4, type=int, metavar='N_S')
group.add_argument('--drop-path', default=0.1, type=float, metavar='drop_path')




def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def plot_loss_and_acc(train_losses, val_losses, out_pth="model.pth"):
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    axs.plot(train_losses)
    axs.plot(val_losses)
    axs.set_yscale('log')
    axs.grid()
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.legend(["Train loss", "Val loss"])
    fig.savefig(out_pth)

def visualize(sample_imgs_unstack, target_imgs_unstack, output_img, outpath = None):
    #visualize 11 sample images in the first row, 11 target images in the second row, and the 11 outputs of the model in the third row
    fig, axs = plt.subplots(3, 11, figsize=(20, 10))
    for i in range(11):
        axs[0, i].imshow(sample_imgs_unstack[i].permute(1, 2, 0))
        axs[1, i].imshow(target_imgs_unstack[i].permute(1, 2, 0))
        axs[2, i].imshow(output_img[i].permute(1, 2, 0))
        axs[0, i].set_title(f"input:{i}")
        axs[1, i].set_title(f"target:{i}")
        axs[2, i].set_title(f"output:{i}")
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[2, i].axis('off')
    plt.tight_layout()
    plt.savefig(outpath)


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
    
    #load pretrained model
    if args.best_model:
        model = torch.load(args.best_model, map_location='cpu')
    else:
        model = SimVP_Model(in_shape=(11,3,160,240), hid_S=args.hid_S, hid_T=args.hid_T, N_T=args.N_T, N_S=args.N_S, drop_path=args.drop_path)

    
    model.to(device)

    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(str(summary(model, input_size=(11, 3,160, 240), batch_dim=0)))
    
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)



    # Dataset
    print("Loading dataset...")
    data_dir = args.data_dir
    train_dataset = UnlabledtrainpredSim(root=os.path.join(data_dir, 'unlabeled'))
    val_dataset = UnlabledtrainpredSim(root=os.path.join(data_dir, 'val'))

    if args.distributed:
        train_sampler = distributed.DistributedSampler(train_dataset)
        val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True, 
        num_workers=args.workers)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        sampler=val_sampler,
        drop_last=True, 
        num_workers=args.workers)


    #loss function
    criterion = MSELoss()

    # train
    print("Starting Training")
    print(f"Training for {args.epochs} epochs ...")

    #main loop
    losses = {'train': [], 'val': []}
    best_loss = 1e9
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        start_time = time.time()
        total_loss = 0

        #train
        model.train()
        print("now training...")
        for x, target in train_dataloader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss/len(train_dataloader)
        print(f"epoch: {epoch:>02}, training_loss: {avg_train_loss:.5f}, training_time:{time.time()-start_time:.2f}")
        scheduler.step()
        losses['train'].append(avg_train_loss)

        #validation
        start_time = time.time()
        print("now validating...")
        model.eval()
        total_loss = 0
        for x, target in val_dataloader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = criterion(output, target)
            total_loss += loss.item()
        avg_val_loss = total_loss/len(val_dataloader)
        ssim = structural_similarity_index_measure(output, target)
        print(f"epoch: {epoch:>02}, ssim: {ssim:.5f}, val_loss: {avg_val_loss:.5f}, validation_time:{time.time()-start_time:.2f}")
        losses['val'].append(avg_val_loss)

        # save best model (only save in validation setting)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if dist.get_rank() == 0:
                torch.save(model_without_ddp.state_dict(), exp_dir + "model_best.pth")
                print("Best model saved with loss: ", best_loss)

        #sample visualize
        if epoch % args.sample_interval == 0:
            model.eval()
            #generate a random number in range of dataset
            sample_imgs, sample_target = val_dataset[random.randint(0, len(val_dataset)-1)]

            #use x to input to model
            x = sample_imgs.to(device)
            x = x.unsqueeze(0) #add batch dim
            output = model(x)
            output = output.detach().cpu().squeeze(0)

            #unbind sequence dim
            sample_imgs_unstack = torch.unbind(sample_imgs, dim=0)
            sample_target_unstack = torch.unbind(sample_target, dim=0)
            output = torch.unbind(output, dim=0)
            visualize(sample_imgs_unstack, sample_target_unstack, output, exp_dir + f'sample_image_on_epoch_{epoch}.pdf')
            print("Sample image saved.")
        
        
        print('-' * 30)
    
    print("Training finished")
    print('-' * 30)

    #write loss to csv
    plot_loss_and_acc(losses['train'], losses['val'], exp_dir + 'loss.pdf')
    data = np.array([losses['train'], losses['val']])
    df_log = pd.DataFrame(data.T, columns=['loss_train', 'loss_val'])
    df_log.to_csv(exp_dir + 'loss.csv', index=False)
                    


    


if __name__ == "__main__":
    main()
