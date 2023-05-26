import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Args:
        img (torch.Tensor): shape (B, C, H, W)
        mean (list or tuple): mean
        std (list or tuple): std
    """
    out = torch.zeros_like(img)
    out[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
    out[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
    out[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
    return out


def cam_on_img(cam, img=None):
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.addWeighted(cam, 0.5, img, 0.5, 0)
    else:
        out = cam

    return out


def get_cam_cmap(cam, img):
    """
    Args:
        cam (torch.Tensor): Valid class activation maps. 4D mini-batch Tensor of shape (B, C, H, W)
        img (torch.Tensor): Original images. 4D mini-batch Tensor of shape (B, 3, H, W)
    """
    cam = cam.detach().cpu()
    img = img.detach().cpu()

    cam_max = cam.max(dim=1)[0].numpy()
    cam_map = plt.get_cmap("jet")(cam_max)[:, :, :, :3] * 255
    cam_map = torch.from_numpy(cam_map).permute((0, 3, 1, 2))

    cam_img = cam_map * 0.5 + img * 0.5
    cam_img = cam_img.to(torch.uint8)
    return cam_img


def minmax_normalize4d(x, eps=1e-6):
    x = F.relu(x)
    x = x + F.adaptive_max_pool2d(-x, (1, 1))
    x = x / (F.adaptive_max_pool2d(x, (1, 1)) + eps)
    return x


def minmax_normalize3d(x, eps=1e-6):
    x = F.relu(x)
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_min = torch.min(x, dim=-1, keepdim=True)[0]
    x = (x - x_min) / (x_max - x_min + eps)
    return x


def minmax_normalize2d(x):
    _size = x.size()
    x = x.flatten(1)
    out = torch.zeros_like(x)
    for b in range(_size[0]):
        out[b] = x[b] - x[b].min()
        out[b] = x[b] / x[b].max()
    out = out.reshape(_size)
    return out


def get_attention_cmap(inputs, normalize=True):
    assert inputs.ndim == 3, "attn should have shape (B, HW, HW)"
    attn = inputs.detach().cpu()
    if normalize:
        attn = minmax_normalize2d(attn)

    cmap = plt.get_cmap("Blues")(attn.numpy())[:, :, :, :3]
    cmap = (cmap * 255).astype(np.uint8)
    cmap = torch.from_numpy(cmap).permute(0, 3, 1, 2).contiguous()
    return cmap


def get_label_cmap(label, size=None):
    if size is not None:
        label = label.to(torch.float32)
        label = F.interpolate(label.unsqueeze(1), size=size, mode="nearest").squeeze(1)
        label = label.to(torch.int64)

    label = label.detach().cpu().numpy()
    label = get_colormap()[label]
    label = torch.from_numpy(label).permute(0, 3, 1, 2).contiguous()
    return label


def get_colormap(N=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap
