import argparse
import os
import os.path as osp

from tools.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dist", type=str2bool, default=False, help="only for cam inference")
parser.add_argument("--eval_freq", type=int, default=1, help="epoch frequency")

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--warmup_epochs", type=int, default=1)

parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--lr_scale", type=float, default=10.0, help="Scale factor for newly added params.")
parser.add_argument("--start_lr", type=float, default=1e-6)
parser.add_argument("--final_lr", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=0.01)

parser.add_argument("--cls_epochs", type=int, default=None)
parser.add_argument("--layer_index", type=int, default=None)
parser.add_argument("--score_l", type=float, default=0.35)
parser.add_argument("--score_h", type=float, default=0.55)
parser.add_argument("--use_ent", action="store_true")

# model settings
parser.add_argument("--arch", type=str, default="sm", choices=["ti", "sm", "bs"])
parser.add_argument("--pretrained", type=str, default="pretrained/conformer_sm.pth")
parser.add_argument("--head_type", type=str, default="class", choices=["class", "patch"])
parser.add_argument("--pool_type", type=str, default="gap", choices=["gap", "gmp"])

# dataset settings
parser.add_argument("--dataset", type=str, default="voc", choices=["voc", "coco"])
parser.add_argument("--data_dir", type=str, default="")
parser.add_argument("--img_dir", type=str, default="")
parser.add_argument("--seg_dir", type=str, default="")
parser.add_argument("--name_list_dir", type=str, default="")
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--crop_size", type=int, default=384)

# inference parameters
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--resize", type=int, default=None)
parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
parser.add_argument("--cam_npy_dir", type=str, default=None)
parser.add_argument("--cam_png_dir", type=str, default=None)

args = parser.parse_args()
if args.dataset == "coco":
    args.data_dir = osp.expanduser("~/data/COCO")
    args.img_dir = osp.expanduser("~/data/COCO/JPEGImages")
    args.seg_dir = osp.expanduser("~/data/COCO/SegmentationClass")
    args.name_list_dir = "data/coco"
    args.num_classes = 80
else:
    args.data_dir = osp.expanduser("~/data/VOCdevkit/VOC2012")
    args.img_dir = osp.expanduser("~/data/VOCdevkit/VOC2012/JPEGImages")
    args.seg_dir = osp.expanduser("~/data/VOCdevkit/VOC2012/SegmentationClass")
    args.name_list_dir = "data/voc"
    args.num_classes = 20
