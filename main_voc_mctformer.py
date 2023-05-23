import argparse
import datetime
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from sklearn.metrics import average_precision_score
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from tqdm import tqdm

from datasets.datasets import VOC12Dataset, VOC12DatasetMS, build_transform
from losses import get_aff_loss
from networks import mctformer
from networks.PAR import PAR
from tools import camutils, imutils, utils

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=str, default="results")

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=0, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")

# Model parameters
parser.add_argument("--model", type=str, default="deit_small_MCTformerV2_patch16_224")
parser.add_argument("--num_classes", type=int, default=20)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--drop", type=float, default=0.0)
parser.add_argument("--drop-path", type=float, default=0.1)

# Optimizer parameters
parser.add_argument("--opt", default="adamw", type=str)
parser.add_argument("--opt_eps", default=1e-8, type=float)
parser.add_argument("--opt_betas", default=None, type=float, nargs="+")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.05)

# Learning rate schedule parameters
parser.add_argument("--sched", default="cosine", type=str)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lr_noise", type=float, nargs="+", default=None)
parser.add_argument("--lr_noise_pct", type=float, default=0.67)
parser.add_argument("--lr_noise_std", type=float, default=1.0)
parser.add_argument("--warmup_lr", type=float, default=1e-6)
parser.add_argument("--min_lr", type=float, default=1e-5)

parser.add_argument("--decay_epochs", type=float, default=30)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--cooldown_epochs", type=int, default=10)
parser.add_argument("--patience_epochs", type=int, default=10)
parser.add_argument("--decay_rate", type=float, default=0.1)

# Augmentation parameters
parser.add_argument("--color_jitter", type=float, default=0.4)
parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1"),
parser.add_argument("--train_interpolation", type=str, default="bicubic")

# Random Erase params
parser.add_argument("--reprob", type=float, default=0.25)
parser.add_argument("--remode", type=str, default="pixel")
parser.add_argument("--recount", type=int, default=1)
parser.add_argument("--resplit", action="store_true")

parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)

# Dataset parameters
parser.add_argument("--data_dir", type=str, default="~/data/VOCdevkit/VOC2012")
parser.add_argument("--name_list_dir", type=str, default="data/voc")
parser.add_argument("--gen_attention_maps", action="store_true")
parser.add_argument("--split", type=str, default="train", choices=["train", "train_aug", "val"])
parser.add_argument("--cam_npy_dir", type=str, default=None)
parser.add_argument("--cam_png_dir", type=str, default=None)

parser.add_argument("--cls_epochs", type=int, default=None)
parser.add_argument("--score_l", type=float, default=0.35)
parser.add_argument("--score_h", type=float, default=0.55)
parser.add_argument("--layer_index", type=int, default=None)
parser.add_argument("--use_ent", action="store_true")


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parser.parse_args()
    print(vars(args))

    fix_random_seed(42)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # build model
    print("model name:", args.model)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    if args.pretrained:
        state_dict = torch.hub.load_state_dict_from_url(args.pretrained, map_location="cpu", check_hash=True)
        state_dict = state_dict["model"]

        model_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in state_dict and state_dict[k].shape != model_dict[k].shape:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        # handle position embedding
        pos_embed = state_dict["pos_embed"]
        num_patches = model.patch_embed.num_patches

        ori_size = int((pos_embed.shape[1] - 1) ** 0.5)
        new_size = int(num_patches ** 0.5)
        print("ori size:", ori_size)
        print("new size:", new_size)

        ext_tokens = pos_embed[:, :1, :].repeat(1, args.num_classes, 1)
        pos_tokens = pos_embed[:, 1:, :]
        if ori_size != new_size:
            b, _, d = pos_tokens.shape
            pos_tokens = pos_tokens.reshape(b, ori_size, ori_size, d).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        state_dict["pos_embed"] = torch.cat((ext_tokens, pos_tokens), dim=1)

        # handle class token
        state_dict["cls_token"] = state_dict["cls_token"].repeat(1, args.num_classes, 1)

        msg = model.load_state_dict(state_dict, strict=False)
        print("finetune from pretrained checkpoint")
        print(msg)

    # build dataset
    if args.gen_attention_maps:
        transform = build_transform(is_train=False, args=args)
        print("transform:", transform)

        dataset = VOC12DatasetMS(
            img_dir=osp.expanduser(osp.join(args.data_dir, "JPEGImages")),
            name_list_path=osp.join(args.name_list_dir, args.split + ".txt"),
            transform=transform,
            scales=[1.0],
        )
        print("dataset size:", len(dataset))

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
        print("data loader size:", len(data_loader))
    else:
        transform_train = build_transform(is_train=True, args=args)
        transform_val = build_transform(is_train=False, args=args)

        dataset_train = VOC12Dataset(
            img_dir=osp.expanduser(osp.join(args.data_dir, "JPEGImages")),
            name_list_path=osp.join(args.name_list_dir, "train_aug.txt"),
            transform=transform_train,
        )
        dataset_val = VOC12Dataset(
            img_dir=osp.expanduser(osp.join(args.data_dir, "JPEGImages")),
            name_list_path=osp.join(args.name_list_dir, "val.txt"),
            transform=transform_val,
        )
        print("dataset size (train):", len(dataset_train))
        print("dataset size (val):", len(dataset_val))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
        )
        print("data loader size (train):", len(data_loader_train))
        print("data loader size (val):", len(data_loader_val))

    if args.gen_attention_maps:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        generate_attention_maps(data_loader, model, args)
        return

    linear_scaled_lr = args.lr * args.batch_size / 512
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    model.cuda()

    # pixel-adaptive refinement
    par = PAR(dilations=[1, 2, 4, 8, 12, 24], num_iters=10)
    par.cuda()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_ent:
        ent = np.load("ent_voc_mctformer.npy")
        ent = ent - ent.min()
        ent = ent / ent.max()
        print("entropy:", ent.tolist())
        ent = torch.from_numpy(ent).to(torch.float32).cuda()
    else:
        ent = None

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch)
        print_freq = 10

        for images, labels in metric_logger.log_every(data_loader_train, print_freq, header):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            class_logits, patch_logits, cam, att = model(images)

            loss1 = F.multilabel_soft_margin_loss(class_logits, labels)
            loss2 = F.multilabel_soft_margin_loss(patch_logits, labels)
            cls_loss = loss1 + loss2

            if args.cls_epochs is not None and epoch >= args.cls_epochs:  # warmup classification
                # generate multi-scale cam
                with torch.no_grad():
                    msf_cam = camutils.gen_msf_cam(model, images, scales=[1.0], args=args, ent=ent)
                    msf_cam = msf_cam * labels.unsqueeze(-1).unsqueeze(-1)

                    # cam to refined seg label
                    images_d = imutils.denormalize(images)
                    seg_labels_l = camutils.cam_to_seg(par, images_d, labels, msf_cam, args.score_l, img_box=None)
                    seg_labels_h = camutils.cam_to_seg(par, images_d, labels, msf_cam, args.score_h, img_box=None)

                    seg_labels = seg_labels_h.clone()
                    seg_labels[(seg_labels_h == 0)] = 255
                    seg_labels[(seg_labels_h + seg_labels_l) == 0] = 0

                    # seg label to aff label
                    aff_labels = camutils.seg_to_aff(seg_labels)

                C = args.num_classes
                aff_loss_list = []
                for i in range(args.layer_index, att.shape[0]):
                    aff_logits = att[i].mean(1)[:, C:, C:]
                    aff_logits = torch.sigmoid(aff_logits)
                    aff_loss = get_aff_loss(aff_logits, aff_labels)
                    aff_loss_list.append(aff_loss)
                aff_loss = torch.stack(aff_loss_list)
                if args.use_ent:
                    aff_loss = (aff_loss * ent[args.layer_index :]).mean()
                else:
                    aff_loss = aff_loss.mean()
                loss = cls_loss + 0.1 * aff_loss
            else:
                aff_loss = torch.tensor(0.0)
                loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            metric_logger.update(cls_loss=cls_loss.item())
            metric_logger.update(aff_loss=aff_loss.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # end of for loop (iteration)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # adjust learning rate
        lr_scheduler.step(epoch)

        val_stats = evaluate(data_loader_val, model)
        print(f"mAP: {val_stats['mAP']*100:.1f}%")

        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state_dict, osp.join(args.output_dir, "checkpoint.pth"))
        if val_stats["mAP"] > max_accuracy:
            torch.save(state_dict, osp.join(args.output_dir, "checkpoint_best.pth"))
        max_accuracy = max(max_accuracy, val_stats["mAP"])
        print(f"Max mAP: {max_accuracy * 100:.2f}%")

        msg = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        with open(osp.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(msg) + "\n")

    # end of for loop (epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


@torch.no_grad()
def evaluate(data_loader, model):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for images, labels in metric_logger.log_every(data_loader, 10, header):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        batch_size = images.shape[0]

        class_logits, patch_logits = model(images)[:2]

        loss1 = F.multilabel_soft_margin_loss(class_logits, labels)
        loss2 = F.multilabel_soft_margin_loss(patch_logits, labels)
        loss = loss1 + loss2

        mAP_list = compute_mAP(torch.sigmoid(class_logits), labels)

        metric_logger.meters["mAP"].update(np.mean(mAP_list), n=batch_size)
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print(f"* mAP {metric_logger.mAP.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(logits, labels):
    y_true = labels.cpu().numpy()
    y_pred = logits.cpu().numpy()
    score_list = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            score = average_precision_score(y_true[i], y_pred[i])
            score_list.append(score)
    return score_list


@torch.no_grad()
def generate_attention_maps(data_loader, model, args):
    model.cuda()
    model.eval()

    if args.cam_npy_dir:
        os.makedirs(args.cam_npy_dir, exist_ok=True)
    if args.cam_png_dir:
        os.makedirs(args.cam_png_dir, exist_ok=True)

    C = args.num_classes
    for name, size, images, labels in tqdm(data_loader):
        name = name[0]
        ori_h, ori_w = size[0].item(), size[1].item()

        key = torch.nonzero(labels[0])[:, 0].numpy()
        if len(key) == 0:
            print(name + " no foreground objects")
            continue

        cam_list = []
        for i, img in enumerate(images):
            img_h = img.shape[2]
            img_w = img.shape[3]
            pad_h = int(np.ceil(img_h / 16) * 16)
            pad_w = int(np.ceil(img_w / 16) * 16)
            pad = (0, pad_w - img_w, 0, pad_h - img_h)
            img = F.pad(img, pad=pad, mode="constant")
            img = img.cuda(non_blocking=True)

            _, _, cam, att = model(img)
            att = att.softmax(dim=-1)  # softmax
            att = att.mean(2)

            # class-to-patch attention refinemnet
            b, c, h, w = cam.shape
            cls_att = att.mean(0)[:, :C, C:].reshape(b, c, h, w)
            cam = cam * cls_att

            cam = cam[:, key, :, :]  # valid cam

            # patch-to-patch attention refinement
            pat_att = att.mean(0)[:, C:, C:]
            b, c, h, w = cam.shape
            cam = torch.einsum("bmn,bcn->bcm", pat_att, cam.reshape(b, c, -1)).reshape(b, c, h, w)

            cam = F.interpolate(cam, size=[pad_h, pad_w], mode="bilinear", align_corners=False)
            cam = cam[:, :, :img_h, :img_w]
            if i % 2 == 1:
                cam = torch.flip(cam, [-1])
            cam = F.interpolate(cam, size=[ori_h, ori_w], mode="bilinear", align_corners=False)
            cam_list.append(cam)

        cam = torch.stack(cam_list, dim=0).sum(dim=0)
        cam = imutils.minmax_normalize4d(cam)
        cam = cam[0].cpu().numpy()

        obj = {"key": key, "cam": cam}
        np.save(osp.join(args.cam_npy_dir, name + ".npy"), obj)

        if args.cam_png_dir:
            path = osp.expanduser(osp.join(args.data_dir, "JPEGImages", name + ".jpg"))
            ori_img = np.array(Image.open(path).convert("RGB"))
            for _key, _cam in zip(key, cam):
                arr = imutils.cam_on_img((_cam * 255).astype(np.uint8), ori_img)
                Image.fromarray(arr).save(osp.join(args.cam_png_dir, f"{name}_cls{_key}.png"))
    # end of for loop (data loader)


if __name__ == "__main__":
    main()
