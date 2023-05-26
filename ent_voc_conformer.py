import argparse
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.datasets import VOC12Dataset
from networks.conformer import conformer_sm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="~/data/VOCdevkit/VOC2012")
parser.add_argument("--name_list_dir", type=str, default="data/voc")


def main():
    args = parser.parse_args()
    device = torch.device("cuda")

    model = conformer_sm(num_classes=20)
    state_dict = torch.load(args.path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    print("Loaded checkpoint.")

    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    dataset = VOC12Dataset(
        img_dir=osp.expanduser(osp.join(args.data_dir, "JPEGImages")),
        name_list_path=osp.join(args.name_list_dir, "train_aug.txt"),
        transform=transform,
    )
    data_loader = DataLoader(dataset, batch_size=1)

    ent_list = []
    for _ in range(12):
        ent_list.append(0.0)

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            _, _, cam, att = model(images)

        att = att.softmax(dim=-1)
        att = att.mean(2)

        valid_labels = torch.nonzero(labels[0])[:, 0].numpy()

        cam = cam[:, valid_labels, :, :]  # valid cam

        for index in range(12):
            # patch-to-patch attention refinement
            pat_att = att[index][:, 1:, 1:]
            b, c, h, w = cam.shape
            new_cam = torch.einsum("bmn,bcn->bcm", pat_att, cam.reshape(b, c, -1))

            # compute entropy-based weights
            prob = new_cam[0].sum(0).softmax(0)
            ent = -torch.sum(prob * torch.log2(prob + 1e-6))
            if torch.isnan(ent).any():
                print("NaN", index)
                continue
            ent_list[index] += ent.item()

    ent = np.array(ent_list) / len(dataset)
    print(ent)
    np.save("ent_voc_conformer.npy", ent)


if __name__ == "__main__":
    main()
