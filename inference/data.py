from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class ImagesToMaskDataset(Dataset):
    def __init__(self, root='../../dataset/hidden/', transform=None):
        self.root = root
        self.transform = transform
        self.vid_list = sorted(os.listdir(root))
        self.img_list = ['image_' + str(i) + '.png' for i in range(11)]
                    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        imgs = []
        for img_idx in self.img_list:
            img_path = os.path.join(self.root, self.vid_list[idx], img_idx)
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)

        if self.transform is not None:
            imgs = self.transform(imgs)

        return torch.stack(imgs)



