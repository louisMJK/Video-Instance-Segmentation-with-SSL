from typing import Optional, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class Transform(MultiViewTransform):
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.5,
        cj_strength: float = 1.0,
        cj_bright: float = 0.1,
        cj_contrast: float = 0.1,
        cj_sat: float = 0.1,
        cj_hue: float = 0.0,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.2,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.5,
        hf_prob: float = 0.5,
        rr_prob: float = 0.5,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.5,
        cj_strength: float = 1.0,
        cj_bright: float = 0.1,
        cj_contrast: float = 0.1,
        cj_sat: float = 0.1,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.2,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.5,
        hf_prob: float = 0.5,
        rr_prob: float = 0.5,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=0.2, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=0.3),
            T.RandomApply([color_jitter], p=0.5),
            T.GaussianBlur(kernel_size=(3, 11), sigma=(0.3, 2.5)),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.transform(image)

