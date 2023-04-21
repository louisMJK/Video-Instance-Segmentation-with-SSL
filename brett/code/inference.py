from cProfile import label
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import argparse
from PIL import Image
import pandas as pd
import numpy as np


DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)

idx_to_class = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}


parser = argparse.ArgumentParser(description='PyTorch Transfer Learning Inference')
parser.add_argument('--model-dir', default='../../dataset/', type=str)
parser.add_argument('--data-dir', default='../../dataset/', type=str)
parser.add_argument('--out-dir', default='', type=str)
parser.add_argument('--img-size', type=int, default=224)
parser.add_argument('--img-val-resize', type=int, default=256)
parser.add_argument('--batch-size', type=int, default=128)


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



def main():
    args = parser.parse_args()

    data_dir = args.data_dir

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Inference on 1 device ({device}).')

    # Test Data
    transform_test = transforms.Compose([
        transforms.Resize(args.img_val_resize),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])
    test_dir = data_dir + "test/"
    dataset_test = DatasetTest(test_dir, transform_test)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # model
    model = torch.load(args.model_dir, map_location=torch.device(device))
    model.eval()

    # inference
    labels = np.array([])
    for inputs in loader_test:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = np.concatenate((labels, preds.cpu().numpy()))
    
    class_pred = [idx_to_class[int(idx)] for idx in labels]

    # save
    df_eval = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    df_eval['file'] = os.listdir(test_dir)
    df_eval['species'] = class_pred
    df_eval.to_csv(args.out_dir + "inference.csv", index=False)



if __name__ == "__main__":
    main()

