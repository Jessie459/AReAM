import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm

from densecrf import crf_inference, crf_inference_label


def get_color_palette(N=256):
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


COLOR_PALETTE = get_color_palette()


def infer_crf_npy(name_list, args):
    if args.score is not None:
        for score in args.score:
            out_dir = osp.join(args.out_dir, f"crf_score{score:.2f}")
            os.makedirs(out_dir, exist_ok=True)
    else:
        for alpha in args.alpha:
            out_dir = osp.join(args.out_dir, f"crf_alpha{alpha}")
            os.makedirs(out_dir, exist_ok=True)

    def _job(start, step, verbose=False):
        for index in tqdm(range(start, len(name_list), step), disable=(verbose is False)):
            name = name_list[index]

            img = np.array(Image.open(osp.join(args.img_dir, name + ".jpg")).convert("RGB"))
            img_h, img_w, _ = img.shape

            cam_path = osp.join(args.cam_dir, name + ".npy")
            if not osp.exists(cam_path):
                print(name + " not exists")
                labels = np.zeros((img_h, img_w), dtype=np.uint8)
                if args.score is not None:
                    for score in args.score:
                        out_dir = osp.join(args.out_dir, f"crf_score{score:.2f}")
                        Image.fromarray(labels).save(osp.join(out_dir, name + ".png"))
                else:
                    for alpha in args.alpha:
                        out_dir = osp.join(args.out_dir, f"crf_alpha{alpha}")
                        Image.fromarray(labels).save(osp.join(out_dir, name + ".png"))
                continue
            cam_dict = np.load(cam_path, allow_pickle=True).item()

            if args.score is not None:
                for score in args.score:
                    out_dir = osp.join(args.out_dir, f"crf_score{score:.2f}")

                    cam_pad = np.pad(cam_dict["cam"], ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=score)
                    crf = crf_inference(img, cam_pad, labels=cam_pad.shape[0])  # (1+fg, h, w)
                    crf_all = np.zeros((args.num_classes + 1, crf.shape[1], crf.shape[2]), dtype=crf.dtype)
                    crf_all[0] = crf[0]
                    for i, k in enumerate(cam_dict["key"]):
                        crf_all[k + 1] = crf[i + 1]
                    labels = np.argmax(crf_all, axis=0)
                    Image.fromarray(labels.astype(np.uint8)).save(osp.join(out_dir, name + ".png"))
                    if args.visualize:
                        Image.fromarray(COLOR_PALETTE[labels].astype(np.uint8)).save(osp.join(out_dir, name + "_c.png"))
            else:
                for alpha in args.alpha:
                    out_dir = osp.join(args.out_dir, f"crf_alpha{alpha}")

                    score = np.power(1 - np.max(cam_dict["cam"], axis=0, keepdims=True), alpha)
                    cam_pad = np.concatenate((score, cam_dict["cam"]), axis=0)
                    crf = crf_inference(img, cam_pad, labels=cam_pad.shape[0])  # (1+fg, h, w)
                    crf_all = np.zeros((args.num_classes + 1, crf.shape[1], crf.shape[2]), dtype=crf.dtype)
                    crf_all[0] = crf[0]
                    for i, k in enumerate(cam_dict["key"]):
                        crf_all[k + 1] = crf[i + 1]
                    labels = np.argmax(crf_all, axis=0)
                    Image.fromarray(labels.astype(np.uint8)).save(osp.join(out_dir, name + ".png"))
                    if args.visualize:
                        Image.fromarray(COLOR_PALETTE[labels].astype(np.uint8)).save(osp.join(out_dir, name + "_c.png"))

    p_list = []
    for i in range(args.nproc):
        p = mp.Process(target=_job, args=(i, args.nproc, (i == 0)))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


def infer_crf_png(name_list, args):
    os.makedirs(args.out_dir, exist_ok=True)

    def _job(start, step, verbose=False):
        for index in tqdm(range(start, len(name_list), step), disable=(verbose is False)):
            name = name_list[index]

            img = np.array(Image.open(osp.join(args.img_dir, name + ".jpg")).convert("RGB"))
            seg = np.array(Image.open(osp.join(args.seg_dir, name + ".png")))

            labels = crf_inference_label(img, seg, n_labels=(args.num_classes + 1))
            Image.fromarray(labels.astype(np.uint8)).save(osp.join(args.out_dir, name + ".png"))
            if args.visualize:
                Image.fromarray(COLOR_PALETTE[labels].astype(np.uint8)).save(osp.join(args.out_dir, name + "_c.png"))

    p_list = []
    for i in range(args.nproc):
        p = mp.Process(target=_job, args=(i, args.nproc, (i == 0)))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="voc", choices=["voc", "coco"])
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--cam_dir", type=str, default=None)
    parser.add_argument("--seg_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="npy", choices=["npy", "png"])
    parser.add_argument("--score", type=float, nargs="+", default=None)
    parser.add_argument("--alpha", type=int, nargs="+", default=None)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("-v", "--visualize", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.dataset == "voc":
        args.num_classes = 20
        with open("data/voc/train_aug.txt", mode="r") as f:
            name_list = np.loadtxt(f, dtype=str)
    else:
        args.num_classes = 80
        with open("data/coco/train.txt", mode="r") as f:
            name_list = np.loadtxt(f, dtype=str)
    print(f"name list: {len(name_list)}")

    print(f"CRF inference starts...")
    if args.suffix == "npy":
        infer_crf_npy(name_list, args)
    else:
        infer_crf_png(name_list, args)
    print(f"CRF inference finished.")


if __name__ == "__main__":
    main()
