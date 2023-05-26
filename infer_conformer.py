import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from config import args
from datasets import coco, voc
from networks.conformer import conformer_sm
from tools.imutils import cam_on_img, minmax_normalize4d


def split_dataset(dataset, n_splits):
    return [torch.utils.data.Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _job(process_id, model, dataset):
    if args.dataset == "voc":
        img_dir = args.img_dir
    else:
        if "train" in args.split:
            img_dir = osp.join(args.img_dir, "train2014")
        else:
            img_dir = osp.join(args.img_dir, "val2014")

    data_loader = torch.utils.data.DataLoader(dataset[process_id], batch_size=1, num_workers=args.num_workers)

    with torch.no_grad(), torch.cuda.device(process_id):
        model.cuda()
        model.eval()

        for name, size, images, labels in tqdm(data_loader, disable=(process_id != 0)):
            name = name[0]
            ori_h = size[0].item()
            ori_w = size[1].item()

            key = torch.nonzero(labels[0])[:, 0].numpy()
            if len(key) == 0:
                print(name + " no objects")
                continue
            cam_list = []

            for i, img in enumerate(images):
                img_h = img.shape[2]
                img_w = img.shape[3]
                pad_h = int(np.ceil(img_h / 16) * 16)
                pad_w = int(np.ceil(img_w / 16) * 16)
                pad = (0, pad_w - img_w, 0, pad_h - img_h)
                img = F.pad(img, pad=pad, mode="constant")

                img = torch.cat([img, img.flip(-1)], dim=0)
                img = img.cuda()

                _, _, cam, att = model(img)
                cam = cam[:, key, :, :]  # valid cam

                att = att.softmax(dim=-1)  # softmax
                att = att.mean(2)  # averaged over attention heads

                # patch-to-patch attention refinement
                pat_att = att.mean(0)[:, 1:, 1:]
                b, c, h, w = cam.shape
                cam = torch.einsum("bmn,bcn->bcm", pat_att, cam.reshape(b, c, -1)).reshape(b, c, h, w)

                cam = F.interpolate(cam, size=[pad_h, pad_w], mode="bilinear", align_corners=False)
                cam1 = cam[:1, ...][:, :, :img_h, :img_w]
                cam2 = cam[1:, ...].flip(-1)[:, :, :img_h, :img_w]
                if img_h != ori_h or img_w != ori_w:
                    cam1 = F.interpolate(cam1, size=[ori_h, ori_w], mode="bilinear", align_corners=False)
                    cam2 = F.interpolate(cam2, size=[ori_h, ori_w], mode="bilinear", align_corners=False)
                cam_list.append(cam1)
                cam_list.append(cam2)
                # cam = torch.max(cam[:1], cam[1:].flip(-1))
                # cam = cam[:, :, :img_h, :img_w]
                # cam = F.interpolate(cam, size=[ori_h, ori_w], mode="bilinear", align_corners=False)
                # cam_list.append(cam)

            cam = torch.stack(cam_list, dim=0).sum(dim=0)
            cam = minmax_normalize4d(cam)
            cam = cam.squeeze(0).cpu().numpy()

            if args.cam_npy_dir:
                obj = {"key": key, "cam": cam}
                np.save(osp.join(args.cam_npy_dir, name + ".npy"), obj)

            if args.cam_png_dir:
                img = np.array(Image.open(osp.join(img_dir, name + ".jpg")).convert("RGB"))
                for _key, _cam in zip(key, cam):
                    arr = cam_on_img((_cam * 255).astype(np.uint8), img)
                    Image.fromarray(arr).save(osp.join(args.cam_png_dir, f"{name}_cls{_key}.png"))
        # end of for loop (data loader)


def main():
    model = conformer_sm(num_classes=args.num_classes)

    print(f"Loading state dict from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    if args.dataset == "voc":
        dataset = voc.VOCClsDatasetMS(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split=args.split,
            resize=args.resize,
            scales=args.scales,
            to_tensor=True,
            normalize=True,
        )
    else:
        dataset = coco.COCOClsDatasetMS(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split=args.split,
            resize=args.resize,
            scales=args.scales,
            to_tensor=True,
            normalize=True,
        )
    print(f"dataset length: {len(dataset)}")

    if args.cam_npy_dir:
        os.makedirs(args.cam_npy_dir, exist_ok=True)
    if args.cam_png_dir:
        os.makedirs(args.cam_png_dir, exist_ok=True)

    print("cam inference starts...")
    print(f"dist: {args.dist}")
    if args.dist:
        n_gpus = torch.cuda.device_count()
        datasets = split_dataset(dataset, n_gpus)
        torch.multiprocessing.spawn(_job, nprocs=n_gpus, args=(model, datasets), join=True)
    else:
        datasets = [dataset]
        _job(0, model, datasets)
    print(f"cam inference finished")


if __name__ == "__main__":
    main()
