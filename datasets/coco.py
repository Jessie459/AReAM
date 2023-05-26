import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

import datasets.transforms as transforms


def robust_read_image(path):
    img = Image.open(path).convert("RGB")
    img_arr = np.array(img)
    if len(img_arr.shape) < 3:
        img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
        img = Image.fromarray(img_arr)
    return img


class COCOClsDataset(Dataset):
    def __init__(self, img_dir, name_list_dir, split, transform=True, crop_size=384):
        if "train" in split:
            self.img_dir = osp.join(img_dir, "train2014")
        else:
            self.img_dir = osp.join(img_dir, "val2014")
        self.crop_size = crop_size
        self.transform = transform
        self.create_transform()

        self.name_list = np.loadtxt(osp.join(name_list_dir, split + ".txt"), dtype=str)
        self.cls_label_dict = np.load(osp.join(name_list_dir, "cls_labels.npy"), allow_pickle=True).item()

    def __len__(self):
        return len(self.name_list)

    def create_transform(self):
        _size = [self.crop_size, self.crop_size]
        self.resize_img = T.Resize(_size, interpolation=T.InterpolationMode.BICUBIC)
        self.resize_seg = T.Resize(_size, interpolation=T.InterpolationMode.NEAREST)

        self.random_scale = transforms.RandomScale(min_scale=0.5, max_scale=2.0)
        self.random_hflip = transforms.RandomHFlip(p=0.5)
        self.color_jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)
        self.random_crop = transforms.RandomCrop(crop_size=self.crop_size, mean_rgb=[0, 0, 0])

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        name = self.name_list[index]

        img = robust_read_image(os.path.join(self.img_dir, name + ".jpg"))

        label = self.cls_label_dict[name + ".jpg"]
        label = torch.from_numpy(label)

        if self.transform is True:
            img = self.random_scale(img)
            img = self.random_hflip(img)
            img = self.color_jitter(img)
            img = np.array(img)
            img, img_box = self.random_crop(img)
            img = self.to_tensor(img)
            img = self.normalize(img)

            return img, label, img_box
        else:
            img = self.resize_img(img)
            img = self.to_tensor(img)
            img = self.normalize(img)

            return img, label


class COCOClsDatasetMS(Dataset):
    def __init__(self, img_dir, name_list_dir, split, resize, scales, to_tensor=True, normalize=True):
        if "train" in split:
            self.img_dir = osp.join(img_dir, "train2014")
        else:
            self.img_dir = osp.join(img_dir, "val2014")

        path = osp.join(name_list_dir, split + ".txt")
        self.name_list = np.loadtxt(path, dtype=str)

        path = osp.join(name_list_dir, "cls_labels.npy")
        self.cls_label_dict = np.load(path, allow_pickle=True).item()

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

        image = robust_read_image(os.path.join(self.img_dir, name + ".jpg"))
        label = self.cls_label_dict[name + ".jpg"]

        size = (image.height, image.width)

        image_list = []
        for s in self.scales:
            if self.resize is not None:
                _size = [int(self.resize * s), int(self.resize * s)]
            else:
                _size = [int(image.width * s), int(image.height * s)]
            _image = image.resize(_size, resample=Image.Resampling.BICUBIC)  # (width, height)
            image_list.append(_image)

        for i in range(len(image_list)):
            if self.to_tensor:
                image_list[i] = self.to_tensor(image_list[i])
            if self.normalize:
                image_list[i] = self.normalize(image_list[i])

        return name, size, image_list, label
