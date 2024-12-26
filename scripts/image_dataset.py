from typing import List, Dict, Tuple
import random
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


class RandomApply:
    def __init__(self, transform: transforms.Compose, p: float) -> None:
        self.transform = transform
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return self.transform(img)
        return img


def generate_train_transforms(image_size: int) -> Tuple[A.Compose, transforms.Compose]:
    train_transform_p1 = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(limit=0.2, p=0.75),
        A.RandomContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        A.Normalize()
    ])
    train_transform_p2 = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomApply(transforms.RandomRotation(45), p=0.5),
        RandomApply(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), p=0.5),
        RandomApply(transforms.GaussianBlur(kernel_size=3), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform_p1, train_transform_p2


def generate_valid_transforms(image_size: int) -> Tuple[A.Compose, transforms.Compose]:
    valid_transform_p1 = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])
    valid_transform_p2 = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return valid_transform_p1, valid_transform_p2


class CustomDatasetP1(Dataset):
    def __init__(self, imgs: List[Image.Image], labels: np.array, transform: A.Compose) -> None:
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.array]:
        imgs_transformed = self.transform(image=np.array(self.imgs[idx]))['image'].astype(np.float32)
        imgs_transformed = imgs_transformed.transpose(2, 0, 1)
        return imgs_transformed, self.labels[idx]


class CustomDatasetP2(Dataset):
    def __init__(self, imgs: List[Image.Image], labels: np.array, transform: transforms.Compose) -> None:
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.array]:
        imgs_transformed = self.transform(self.imgs[idx])
        return imgs_transformed, self.labels[idx]