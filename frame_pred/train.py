import torch
from torch.nn import MSELoss
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
import copy
import pandas as pd
import matplotlib.pyplot as plt

from optimizer import create_optimizer
from utils import Unlabledtrainpred, Predictor


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--num-filter', default='9', type=int, metavar='FILTER')
group.add_argument('--kernel-size', default='3', type=int, metavar='KERNEL')
group.add_argument('--width', default='240', type=int, metavar='WIDTH')
group.add_argument('--height', default='160', type=int, metavar='HEIGHT')

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
group.add_argument('-b', '--batch-size', type=int, default=64, metavar='N')
group.add_argument('--workers', type=int, default=4, metavar='N')
group.add_argument('--inference', action='store_true', default=True)
group.add_argument('--data-dir', default='dataset/', type=str)
group.add_argument('--out-dir', default='../../output/', type=str)
group.add_argument('--verbose', action='store_true', default=False)
group.add_argument('--val-interval', type=int, default=2)



def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def plot_loss_and_acc(train_losses, val_losses, out_pth="model.pth"):
    fig, axs = plt.subplots(1, 1, figsize=(16, 6))
    axs[0].plot(train_losses)
    axs[0].plot(val_losses)
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(["Train loss", "Val loss"])
    fig.savefig(out_pth)

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
    model = Predictor(3, 9, (args.batch_size, args.height, args.width), args.kernel_size, device).to(device)


    # with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
    #     f.write(str(model))
    #     f.write('\n\n')
    #     f.write(str(summary(model, input_size=(11, 3, args.height, args.width), batch_dim=0)))


    # optimizer
    optimizer = create_optimizer(model, args)

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
    train_dataset = Unlabledtrainpred(root=os.path.join(data_dir, 'unlabeled'))
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=args.workers)
    val_dataset = Unlabledtrainpred(root=os.path.join(data_dir, 'val'))
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
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
        start_time = time.time()
        total_loss = 0

        #train
        model.train()
        for x, target in train_dataloader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_dataloader)
        print(f"epoch: {epoch:>02}, training_loss: {avg_loss:.5f}, training_time:{time.time()-start_time:.2f}")
        scheduler.step()
        losses['train'].append(avg_loss)

        #validation
        if epoch % args.val_interval == 0:
            start_time = time.time()
            model.eval()
            total_loss = 0
            for x, target in val_dataloader:
                x = x.to(device)
                target = target.to(device)
                output = model(x)
                loss = criterion(output, target)
                total_loss += loss.item()
            avg_loss = total_loss/len(val_dataloader)
            print(f"epoch: {epoch:>02}, val_loss: {avg_loss:.5f}, validation_time:{time.time()-start_time:.2f}")
            losses['val'].append(avg_loss)

        #save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, exp_dir + "model_best.pth")
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model saved with loss: ", best_loss)
        
        model.load_state_dict(best_model_wts)
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

