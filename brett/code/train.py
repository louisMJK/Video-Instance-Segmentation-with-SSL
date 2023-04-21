import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchinfo import summary

import os
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import argparse
from PIL import Image

from torchvision.models import resnet50, vit_b_16
from optimizer import create_optimizer
from optimizer import create_optimizer


DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet18', type=str, metavar='MODEL')
group.add_argument('--img-size', type=int, default=224)
group.add_argument('--img-val-resize', type=int, default=256)
group.add_argument('--layers-unfreeze', type=int, default=0)
group.add_argument('--fine-tune', type=int, default=0)

# Optimizer & Scheduler parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
group.add_argument('--optim', default='adam', type=str, metavar='OPTIMIZER')
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--lr-base', type=float, default=1e-3, metavar='LR')
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
    if args.model == 'resnet50':
        model = resnet50(args.layers_unfreeze, args.fine_tune)
    elif args.model == 'vit_b_16':
        # img_size = 384, img-val-resize = 384
        model = vit_b_16(args.layers_unfreeze, args.fine_tune)
    else:
        model = resnet50()
    model.to(device=device)

    with open(os.path.join(exp_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(str(summary(model, input_size=(3, args.img_size, args.img_size), batch_dim=0)))


    # optimizer
    optimizer = create_optimizer(model, args)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

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


    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(180),
        # transforms.RandomInvert(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.ColorJitter(
        #     brightness=0.4,
        #     contrast=0.4,
        #     saturation=0.4,
        # ),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(args.img_val_resize),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])

    # Dataset
    print("Loading dataset...")
    data_dir = args.data_dir
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_val)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    val_size = int(0.2 * dataset_size)
    dataset_train = Subset(dataset, indices[: -val_size])
    dataset_val = Subset(dataset_val, indices[-val_size: ])

    # Dataloaders
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dataloaders = {'train': loader_train, 'val': loader_val}
    dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}

    # train
    print()
    print(f"Training {args.model} for {args.epochs} epochs ...")
    model_best, losses, accs = \
        train_model(model, loss_fn, optimizer, scheduler, dataloaders, dataset_sizes, args.epochs, device)
    
    # save best model
    torch.save(model_best, exp_dir + "model_" + str(max(accs['val'])) + '.pth')
    print("Best model saved.")
    print('-' * 30)
    
    # make plot and log
    plot_loss_and_acc(losses['train'], accs['train'], losses['val'], accs['val'], exp_dir + 'loss_acc.pdf')
    data = np.array([losses['train'], losses['val'], accs['train'], accs['val']])
    df_log = pd.DataFrame(data.T, columns=['loss_train', 'loss_val', 'acc_train', 'acc_val'])
    df_log.to_csv(exp_dir + "log.csv", index=False)

    # inferece
    if args.inference:
        test_dir = data_dir + "test/"
        dataset_test = DatasetTest(test_dir, transform_val)
        loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        class_pred = inference(model_best, loader_test, dataset.class_to_idx, device)

        df_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
        df_submission['file'] = os.listdir(test_dir)
        df_submission['species'] = class_pred
        df_submission.to_csv(exp_dir + "inference.csv", index=False)
        print('Inference completed.')



def train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        dataloaders, 
        dataset_sizes, 
        num_epochs = 10,  
        device = 'cuda'
):
    model.to(device)
    t_start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = {'train': [], 'val': []}
    accs = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        t1 = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            if phase == 'train':
                print(f'Epoch [{epoch+1:4d}/{num_epochs:4d}]\t{phase} Loss: {epoch_loss:.3e}, Acc: {epoch_acc:.4f},', end='')
            else:
                epoch_time = time.time() - t1
                print(f'\t{phase} Loss: {epoch_loss:.3e}, Acc: {epoch_acc:.4f}   Time: {epoch_time:.0f}s')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - t_start
    print(f'Training completed in {time_elapsed // 60:.0f} min {time_elapsed % 60:.0f} s')
    print(f'Best Validation Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, losses, accs



class DatasetTest(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    


def inference(model, loader_test, class_to_idx, device='cuda'):
    # load model
    model.to(device)
    model.eval()
    labels = np.array([])

    # inference
    for inputs in loader_test:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = np.concatenate((labels, preds.cpu().numpy()))
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_pred = [idx_to_class[int(idx)] for idx in labels]

    return class_pred



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
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(["Train acc", "Val acc"])
    axs[1].set_ylim([0, 1.0])
    
    fig.savefig(out_pth)



if __name__ == "__main__":
    main()

