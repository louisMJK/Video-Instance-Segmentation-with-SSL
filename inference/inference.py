import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import argparse
import yaml
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime

from models.large_model import LargeModel
from data import ImagesToMaskDataset


DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


parser = argparse.ArgumentParser(description='PyTorch Video Instance Segmentation')
parser.add_argument('--predictor-dir', default='', type=str)
parser.add_argument('--fcn-dir', default='', type=str)
parser.add_argument('--data-dir', default='../dataset/', type=str)

parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4, metavar='N')
parser.add_argument("--dist-url", default="env://", type=str)
parser.add_argument("--world-size", default=1, type=int)

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
    print(f'\nTraining on {device}.')

    # Data
    transform_test =T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])
    data_dir = args.data_dir
    hidden_dataset = ImagesToMaskDataset(os.path.join(data_dir, 'hidden'), transform_test)
    hidden_loader = DataLoader(hidden_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    hidden_size = len(hidden_dataset)

    # model
    model = LargeModel()
    model.predictor.load_state_dict(torch.load(args.predictor_dir, map_location='cpu'), strict=True)
    model.fcn_resnet.load_state_dict(torch.load(args.fcn_dir, map_location='cpu'), strict=True)
    model.to(device)


    # inference
    print('Start Inference ...')
    model.eval()
    masks = np.zeros((0, 160, 240))

    for i, x in enumerate(hidden_loader):
        x = x.to(device)
        with torch.no_grad():
            y = model(x)
            out = y['out']
        masks_batch = out.cpu().detach().numpy().argmax(1)
        masks = np.concatenate([masks, masks_batch], axis=0)
        if (i+1) % 20 == 0:
            print(f'Batch {i+1} / {len(hidden_loader)}')

    assert masks.shape == (hidden_size, 160, 240)

    # save
    np.save(exp_dir + 'masks.npy', masks)



if __name__ == "__main__":
    main()

