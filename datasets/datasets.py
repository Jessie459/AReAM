import os

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import Dataset
from torchvision import transforms


class VOC12Dataset(Dataset):
    def __init__(self, img_dir, name_list_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.name_list = np.loadtxt(name_list_path, dtype=str)
        self.cls_labels = np.load("data/voc/cls_labels.npy", allow_pickle=True).item()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        if self.transform:
            img = self.transform(img)

        cls_label = self.cls_labels[name].astype(int)
        cls_label = torch.from_numpy(cls_label)

        return img, cls_label


class VOC12DatasetMS(Dataset):
    def __init__(self, img_dir, name_list_path, scales, transform=None, unit=1):
        self.img_dir = img_dir
        self.name_list = np.loadtxt(name_list_path, dtype=str)
        self.cls_labels = np.load("data/voc/cls_labels.npy", allow_pickle=True).item()
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]

        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        size = (img.height, img.width)

        cls_label = self.cls_labels[name].astype(int)
        cls_label = torch.from_numpy(cls_label)

        rounded_size = (
            int(round(img.size[0] / self.unit) * self.unit),
            int(round(img.size[1] / self.unit) * self.unit),
        )

        ms_img_list = []
        for s in self.scales:
            _size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
            ms_img = img.resize(_size, resample=Image.CUBIC)
            ms_img_list.append(ms_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))

        return name, size, msf_img_list, cls_label


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        )  # to maintain same ratio w.r.t. 224 images
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return transforms.Compose(t)
