import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import os
import argparse
import yaml
import numpy as np
from datetime import datetime

from models.main_model import MainModel
from data import HiddenDataset, ImagesToMaskDataset


parser = argparse.ArgumentParser(description='PyTorch Video Instance Segmentation')
parser.add_argument('--model-dir', default='', type=str)
parser.add_argument('--data-dir', default='../dataset/', type=str)
parser.add_argument('--val', action='store_true', default=False)

parser.add_argument('-b', '--batch-size', type=int, default=4)
parser.add_argument('--workers', type=int, default=4)


def _parse_args():
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text



def main():
    args, args_text = _parse_args()
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S")])
    exp_dir = exp_name + "/"
    os.mkdir(exp_dir)

    print(args)

    # logging
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'\nDevice: {device}.')

    # Data
    data_dir = args.data_dir
    hidden_dataset = HiddenDataset(os.path.join(data_dir, 'hidden'))
    val_dataset = ImagesToMaskDataset(os.path.join(data_dir, 'val'))

    hidden_loader = DataLoader(hidden_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    hidden_size = len(hidden_dataset)


    # model
    model = MainModel()
    model.load_state_dict(torch.load(args.model_dir, map_location='cpu'), strict=True)
    model.to(device)
    print(f'Model loaded from {args.model_dir}')

    # inference
    print('\nRunning Inference ...')
    model.eval()
    masks_test = np.zeros((0, 160, 240))

    for i, x in enumerate(hidden_loader):
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
            out = y['out']
        masks_batch = out.cpu().detach().numpy().argmax(1)
        masks_test = np.concatenate([masks_test, masks_batch], axis=0)
        if (i+1) % 100 == 0:
            print(f'Batch {i+1} / {len(hidden_loader)}')

    assert masks_test.shape == (2000, 160, 240)

    # save
    np.save(exp_dir + 'masks_hidden.npy', masks_test)
    print(f'\nMasks saved. Shape: {masks_test.shape}')


    # validation
    if args.val:
        print('\nRunning Validation ...')
        model.eval()
        masks_val = np.zeros((0, 160, 240))
        masks = np.zeros((0, 160, 240))
        jaccard = JaccardIndex(task="multiclass", num_classes=49)

        with torch.no_grad():
            for (x, targets) in val_loader:
                x = x.to(device)
                y = model(x)
                out = y['out']
                masks_batch = out.cpu().numpy().argmax(1)
                masks_val = np.concatenate([masks_val, masks_batch], axis=0)
                masks = np.concatenate([masks, targets.cpu().numpy()], axis=0)

        jac = jaccard(torch.Tensor(masks_val), torch.Tensor(masks))

        print(f'Validation Jaccard: {jac:.5f}\n')
        np.save(exp_dir + 'masks_' + str(jac) +'.npy', masks_test)



if __name__ == "__main__":
    main()

