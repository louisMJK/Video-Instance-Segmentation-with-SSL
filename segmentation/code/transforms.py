import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class SegmentationTrainTransform:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = 240
        max_size = 300

        trans = [
            RandomResize(min_size, max_size), 
            RandomHorizontalFlip(0.5), 
            RandomVerticalFlip(0.5),
        ]

        trans.extend(
            [
                RandomCrop(),
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationValTransform:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


def pad_if_smaller(img, size=160, fill=0):
    min_size = min(img.size)
    if min_size < size:
        print('-' * 50)
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = target.unsqueeze(0)
        target = F.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
        target = target.squeeze(0)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size=200):
        self.size = size

    def __call__(self, image, target):
        # image = pad_if_smaller(image, self.size)
        # target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(image, (160, 240))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
