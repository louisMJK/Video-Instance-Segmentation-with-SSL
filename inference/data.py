import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np
import os
from PIL import Image

DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


transform_test =T.Compose([
    T.ToTensor(),
    T.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
])


class HiddenDataset(Dataset):
    def __init__(self, root='../dataset/hidden/', transform=transform_test):
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
            img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs)


class SegmentationValTransform:
    def __init__(self, mean=DATA_MEAN, std=DATA_STD):
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    

class PILToTensor:
    def __call__(self, images, target):
        imgs = [F.pil_to_tensor(image) for image in images]
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return imgs, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, images, target):
        imgs = [F.convert_image_dtype(image, self.dtype) for image in images]
        return imgs, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target):
        imgs = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        return imgs, target


class ImagesToMaskDataset(Dataset):
    def __init__(self, root='../dataset/val/', transform=SegmentationValTransform()):
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

