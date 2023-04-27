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

    def __call__(self, imgs, target):
        return self.transforms(imgs, target)


class SegmentationValTransform:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, imgs, target):
        return self.transforms(imgs, target)


def pad_if_smaller(img, size=160, fill=0):
    min_size = min(img.size)
    if min_size < size:
        print('?' * 50)
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        for t in self.transforms:
            images, target = t(images, target)
        return images, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, images, target):
        size = random.randint(self.min_size, self.max_size)
        imgs = [F.resize(img, size) for img in images]
        target = target.unsqueeze(0)
        target = F.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
        target = target.squeeze(0)
        return imgs, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        imgs = images
        if random.random() < self.flip_prob:
            imgs = [F.hflip(image) for image in images]
            target = F.hflip(target)
        return imgs, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, images, target):
        imgs = images
        if random.random() < self.flip_prob:
            imgs = [F.vflip(image) for image in images]
            target = F.vflip(target)
        return imgs, target


class RandomCrop:
    def __init__(self, size=200):
        self.size = size

    def __call__(self, images, target):
        # image = pad_if_smaller(image, self.size)
        # target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(images[0], (160, 240))
        imgs = [F.crop(image, *crop_params) for image in images]
        target = F.crop(target, *crop_params)
        return imgs, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, images, target):
        imgs = [F.center_crop(image, self.size) for image in images]
        target = F.center_crop(target, self.size)
        return imgs, target


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

