from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np


class ImagesToMaskDataset(Dataset):
    def __init__(self, root='../../dataset/train/', transform=None):
        self.root = root
        self.transform = transform
        self.vid_list = sorted(os.listdir(root))
        self.img_list = ['image_' + str(i) + '.png' for i in range(11)]
                    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        # load images
        imgs = []
        for img_idx in self.img_list:
            img_path = os.path.join(self.root, self.vid_list[idx], img_idx)
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)

        # load target mask
        mask_path = os.path.join(self.root, self.vid_list[idx], 'mask.npy')
        target = torch.Tensor(np.load(mask_path)[-1])

        if self.transform is not None:
            imgs, target = self.transform(imgs, target)

        return torch.stack(imgs), target

