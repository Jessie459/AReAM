import random

import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img


class RandomHFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label=None):
        if label is None:
            if random.random() < self.p:
                image = ImageOps.mirror(image)
            return image
        else:
            if random.random() < self.p:
                image = ImageOps.mirror(image)
                label = ImageOps.mirror(label)
            return image, label


class RandomScale(object):
    def __init__(self, min_scale, max_scale):
        assert min_scale <= max_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, label=None):
        scale = random.uniform(self.min_scale, self.max_scale)
        width, height = int(scale * image.width), int(scale * image.height)

        new_image = image.resize([width, height], resample=Image.BICUBIC)
        if label is None:
            return new_image

        new_label = label.resize([width, height], resample=Image.NEAREST)
        return new_image, new_label


def img_resize_short(image, min_size=512):
    h, w, _ = image.shape
    if min(h, w) >= min_size:
        return image

    scale = float(min_size) / min(h, w)
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    return new_image


def random_fliplr(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image = np.fliplr(image)
        return image
    else:
        if p > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        return image, label


def random_flipud(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image = np.flipud(image)
        return image
    else:
        if p > 0.5:
            image = np.flipud(image)
            label = np.flipud(label)

        return image, label


def random_rot(image, label):
    k = random.randrange(3) + 1

    image = np.rot90(image, k).copy()

    if label is None:
        return image

    label = np.rot90(label, k).copy()

    return image, label


class RandomCrop(object):
    def __init__(self, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
        self.crop_size = crop_size
        self.mean_rgb = mean_rgb
        self.ignore_index = ignore_index

    def get_random_box(self, H, W, label=None, cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - self.crop_size + 1, 1)
            H_end = H_start + self.crop_size
            W_start = random.randrange(0, W - self.crop_size + 1, 1)
            W_end = W_start + self.crop_size

            if label is None:
                return (H_start, H_end, W_start, W_end)

            temp_label = label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != self.ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return (H_start, H_end, W_start, W_end)

    def __call__(self, image, label=None):
        img_h, img_w, _ = image.shape

        H = max(self.crop_size, img_h)
        W = max(self.crop_size, img_w)
        pad_image = np.zeros((H, W, 3), dtype=np.uint8)
        pad_image[:, :, 0] = self.mean_rgb[0]
        pad_image[:, :, 1] = self.mean_rgb[1]
        pad_image[:, :, 2] = self.mean_rgb[2]

        H_pad = int(np.random.randint(H - img_h + 1))
        W_pad = int(np.random.randint(W - img_w + 1))
        pad_image[H_pad:(H_pad + img_h), W_pad:(W_pad + img_w), :] = image

        H_start, H_end, W_start, W_end = self.get_random_box(H, W, label=label)
        crop_image = pad_image[H_start:H_end, W_start:W_end, :]

        img_H_start = max(H_pad - H_start, 0)
        img_W_start = max(W_pad - W_start, 0)
        img_H_end = min(self.crop_size, img_h + H_pad - H_start)
        img_W_end = min(self.crop_size, img_w + W_pad - W_start)
        img_box = np.array([img_H_start, img_H_end, img_W_start, img_W_end], dtype=int)

        if label is None:
            return crop_image, img_box

        pad_label = np.ones((H, W), dtype=np.uint8) * self.ignore_index
        pad_label[H_pad:(H_pad + img_h), W_pad:(W_pad + img_w)] = label
        label = pad_label[H_start:H_end, W_start:W_end]

        return crop_image, label, img_box
