import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from . import transforms


class VOCClsDataset(Dataset):
    def __init__(self, img_dir, name_list_dir, split="train_aug", transform=True, crop_size=384):
        self.img_dir = img_dir

        self.crop_size = crop_size
        self.transform = transform
        self._build_transform()

        path = osp.join(name_list_dir, split + ".txt")
        self.name_list = np.loadtxt(path, dtype=str)

        path = osp.join(name_list_dir, "cls_labels.npy")
        self.cls_labels = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.name_list)

    def _build_transform(self):
        self.resize = T.Resize([self.crop_size, self.crop_size], interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.random_scale = transforms.RandomScale(min_scale=0.5, max_scale=2.0)
        self.random_hflip = transforms.RandomHFlip(p=0.5)
        self.color_jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)
        self.random_crop = transforms.RandomCrop(crop_size=self.crop_size, mean_rgb=[0, 0, 0])

    def __getitem__(self, index):
        name = self.name_list[index]

        img = Image.open(osp.join(self.img_dir, name + ".jpg")).convert("RGB")

        cls_label = self.cls_labels[name].astype(int)
        cls_label = torch.from_numpy(cls_label)

        if self.transform is True:
            img = self.random_scale(img)
            img = self.random_hflip(img)
            img = self.color_jitter(img)
            img = np.array(img)
            img, img_box = self.random_crop(img)
            img = self.to_tensor(img)
            img = self.normalize(img)

            return img, cls_label, img_box
        else:
            img = self.resize(img)
            img = self.to_tensor(img)
            img = self.normalize(img)

            return img, cls_label


class VOCClsDatasetMS(Dataset):
    def __init__(self, img_dir, name_list_dir, split, resize, scales, to_tensor=True, normalize=True):
        self.img_dir = img_dir

        path = osp.join(name_list_dir, split + ".txt")
        self.name_list = np.loadtxt(path, dtype=str)

        path = osp.join(name_list_dir, "cls_labels.npy")
        self.cls_labels = np.load(path, allow_pickle=True).item()

        if resize is not None:
            if isinstance(resize, (tuple, list)):
                assert len(resize) == 2, "resize should be a 2-tuple of (width, height)"
            else:
                resize = [resize, resize]
        self.resize = resize
        self.scales = scales

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.to_tensor = T.ToTensor() if to_tensor is True else None
        self.normalize = T.Normalize(mean=mean, std=std) if normalize is True else None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]

        img = Image.open(osp.join(self.img_dir, name + ".jpg")).convert("RGB")
        size = (img.height, img.width)

        cls_label = self.cls_labels[name].astype(int)
        cls_label = torch.from_numpy(cls_label)

        ms_img_list = []
        for s in self.scales:
            if self.resize is not None:
                _size = [int(self.resize[0] * s), int(self.resize[1] * s)]
            else:
                _size = [int(img.width * s), int(img.height * s)]

            ms_img = img.resize(_size, resample=Image.BICUBIC)  # requested size: (width, height)
            ms_img_list.append(ms_img)

        for i in range(len(ms_img_list)):
            if self.to_tensor:
                ms_img_list[i] = self.to_tensor(ms_img_list[i])
            if self.normalize:
                ms_img_list[i] = self.normalize(ms_img_list[i])

        return name, size, ms_img_list, cls_label
