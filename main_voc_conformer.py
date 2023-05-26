import datetime
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import args
from datasets import coco, voc
from losses import get_aff_loss
from networks.conformer import conformer_sm
from networks.PAR import PAR
from optim import cosine_schedule
from tools import camutils, imutils, utils

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main():
    utils.fix_random_seed(args.seed, deterministic=True)

    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    writer_dir = os.path.join(args.output_dir, "runs", timestamp)
    logger_dir = os.path.join(args.output_dir, "logs")

    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)

    logger = utils.get_logger(filename=osp.join(logger_dir, timestamp + ".log"))
    writer = SummaryWriter(writer_dir)

    msg = ""
    for key, val in vars(args).items():
        msg += f"{key}: {val}\n"
    print(msg)
    logger.info(msg)

    # =====================================
    # create the model
    # =====================================
    model = conformer_sm(num_classes=args.num_classes)
    logger.info(str(model))

    state_dict = torch.load(args.pretrained, map_location="cpu")
    for k in list(state_dict.keys()):
        if "head" in k:
            del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded pretrained model.")
    logger.info(f"=> missing keys: {msg.missing_keys}")
    logger.info(f"=> unexpected keys: {msg.unexpected_keys}")

    par = PAR(dilations=[1, 2, 4, 8, 12, 24], num_iters=10)
    par.cuda()

    # =====================================
    # create the dataset
    # =====================================
    if args.dataset == "voc":
        dataset_train = voc.VOCClsDataset(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split="train_aug",
            transform=True,
            crop_size=args.crop_size,
        )
        dataset_val = voc.VOCClsDataset(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split="val",
            transform=False,
            crop_size=args.crop_size,
        )
    else:
        dataset_train = coco.COCOClsDataset(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split="train",
            transform=True,
            crop_size=args.crop_size,
        )
        dataset_val = coco.COCOClsDataset(
            img_dir=args.img_dir,
            name_list_dir=args.name_list_dir,
            split="val",
            transform=False,
            crop_size=args.crop_size,
        )
    logger.info(f"size of dataset (train): {len(dataset_train)}")
    logger.info(f"size of dataset (val): {len(dataset_val)}")

    # =====================================
    # create the data loader
    # =====================================
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, pin_memory=True,)
    logger.info(f"data loader iterations (train): {len(data_loader_train)}")
    logger.info(f"data loader iterations (val): {len(data_loader_val)}")

    # =====================================
    # create the optimizer and scheduler
    # =====================================
    param_groups = get_param_groups(model, weight_decay=args.weight_decay, lr_scale=args.lr_scale)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(data_loader_train) * args.epochs
    warmup_steps = len(data_loader_train) * args.warmup_epochs
    scheduler = cosine_schedule(
        args.lr,
        start_lr=args.start_lr,
        final_lr=args.final_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # =====================================
    # create the criterion
    # =====================================
    cls_criterion = torch.nn.MultiLabelSoftMarginLoss()
    cls_criterion = cls_criterion.cuda()

    # move the model to GPU
    model.cuda()

    # switch to train mode
    model.train()

    start_time = time.time()
    best_cnn_score = 0.0
    best_att_score = 0.0
    logger.info(f"Start training for {args.epochs} epochs.")

    if args.use_ent:
        ent = np.load("ent_voc_conformer.npy")
        ent = ent - ent.min()
        ent = ent / ent.max()
        logger.info("entropy:", ent.tolist())
        ent = torch.from_numpy(ent).to(torch.float32).cuda()
    else:
        ent = None

    global_step = 0
    for epoch in range(args.epochs):
        # =====================================
        # training process
        # =====================================
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        print_freq = 10

        for step, data in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
            # adjust the learning rate
            lr = scheduler[global_step]
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr

            images, labels, img_box = data
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # forward propagation
            cnn_logits, att_logits, cam, att = model(images)

            # classification loss
            cls_loss = cls_criterion(cnn_logits, labels) + cls_criterion(att_logits, labels)

            seg_labels = None
            aff_labels = None
            if args.cls_epochs is not None and epoch >= args.cls_epochs:
                # generate multi-scale cam
                with torch.no_grad():
                    msf_cam = camutils.gen_msf_cam(model, images, scales=[0.5, 1.0, 1.5], args=args, ent=ent)
                    msf_cam = msf_cam * labels.unsqueeze(-1).unsqueeze(-1)

                    # cam to refined seg label
                    images_d = imutils.denormalize(images)
                    seg_labels_l = camutils.cam_to_seg(par, images_d, labels, msf_cam, args.score_l, img_box=img_box)
                    seg_labels_h = camutils.cam_to_seg(par, images_d, labels, msf_cam, args.score_h, img_box=img_box)

                    seg_labels = seg_labels_h.clone()
                    seg_labels[(seg_labels_h == 0)] = 255
                    seg_labels[(seg_labels_h + seg_labels_l) == 0] = 0

                    # seg label to aff label
                    aff_labels = camutils.seg_to_aff(seg_labels)

                aff_loss_list = []
                for i in range(args.layer_index, att.shape[0]):
                    aff_logits = att[i].mean(1)[:, 1:, 1:]
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
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            metric_logger.update(cls_loss=cls_loss.item())
            metric_logger.update(aff_loss=aff_loss.item())

            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/cls_loss", cls_loss.item(), global_step)
            writer.add_scalar("train/aff_loss", aff_loss.item(), global_step)

            if (global_step + 1) % 100 == 0:
                with torch.no_grad():
                    img = images.clone()
                    img[:, 0, :, :] = img[:, 0, :, :] * STD[0] + MEAN[0]
                    img[:, 1, :, :] = img[:, 1, :, :] * STD[1] + MEAN[1]
                    img[:, 2, :, :] = img[:, 2, :, :] * STD[2] + MEAN[2]
                    img = (img * 255).to(torch.uint8)
                    B, _, img_h, img_w = img.shape

                    # log class activation maps
                    arr = F.interpolate(cam, size=[img_h, img_w], mode="bilinear", align_corners=False)
                    arr = imutils.minmax_normalize4d(arr)
                    arr = arr * labels.unsqueeze(2).unsqueeze(3)
                    arr = imutils.get_cam_cmap(arr[:2], img[:2])
                    writer.add_images("cam", arr, global_step=global_step)

                    # log attention maps
                    for i in range(att.shape[0]):
                        arr = att[i].softmax(-1).mean(1)[:, 1:, 1:]
                        writer.add_images(f"att/layer{i}", imutils.get_attention_cmap(arr[:2]), global_step=global_step)

                        arr = torch.einsum("bmn,bcn->bcm", arr, cam.flatten(2)).reshape(cam.shape)
                        arr = F.interpolate(arr, size=img.size()[2:], mode="bilinear", align_corners=False)
                        arr = imutils.minmax_normalize4d(arr)
                        arr = arr * labels.unsqueeze(2).unsqueeze(3)
                        arr = imutils.get_cam_cmap(arr[:2], img[:2])
                        writer.add_images(f"cam_refined/layer{i}", arr, global_step=global_step)

                    if seg_labels is not None:
                        arr = imutils.get_label_cmap(seg_labels)
                        writer.add_images(f"label/seg", arr[:2], global_step=global_step)
                    if aff_labels is not None:
                        arr = imutils.get_label_cmap(aff_labels)
                        writer.add_images(f"label/aff", arr[:2], global_step=global_step)

            global_step += 1
        # end of for loop (data loader)

        metric_logger.synchronize_between_processes()
        msg = "[Train stats]"
        for k, v in metric_logger.meters.items():
            msg += f" {k}: {v.global_avg:.6f}"
        logger.info(msg)

        # save the checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict()}, osp.join(args.output_dir, "checkpoint.pth"))

        # =====================================
        # evaluation process
        # =====================================
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            test_stats = evaluate(data_loader_val, model)

            msg = "[Test stats]"
            for k, v in test_stats.items():
                msg += f" {k}: {v:.4f}"
            logger.info(msg)
            for k, v in test_stats.items():
                writer.add_scalar(f"test/{k}", v, epoch)

            cnn_score = test_stats["cnn_score"]
            att_score = test_stats["att_score"]
            if cnn_score > best_cnn_score:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()}, osp.join(args.output_dir, "checkpoint_best_cnn.pth")
                )
                best_cnn_score = cnn_score
            if att_score > best_att_score:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()}, osp.join(args.output_dir, "checkpoint_best_att.pth")
                )
                best_att_score = att_score

    # end of for loop (epoch)
    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training finished. (time: {total_time_str})")


@torch.no_grad()
def evaluate(data_loader, model):
    # switch to eval mode
    model.eval()

    true_list = []
    pred_list = [[], []]

    for images, labels in tqdm(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        cnn_logits, att_logits, _, _ = model(images)

        true_list.append(labels)
        pred_list[0].append(cnn_logits)
        pred_list[1].append(att_logits)

    score_list = []
    y_true = torch.cat(true_list, dim=0).cpu().numpy()
    for i in range(len(pred_list)):
        y_pred = (torch.cat(pred_list[i], dim=0) > 0).int().cpu().numpy()
        score = f1_score(y_true=y_true, y_pred=y_pred, average="samples")
        score_list.append(score)

    # switch to train mode
    model.train()

    stats = {}
    stats["cnn_score"] = score_list[0]
    stats["att_score"] = score_list[1]
    return stats


def get_param_groups(model, weight_decay=1e-4, lr_scale=1.0):
    if hasattr(model, "no_weight_decay"):
        no_weight_decay = model.no_weight_decay()
    else:
        no_weight_decay = {}

    decay_params = []
    nodec_params = []
    new_decay_params = []
    new_nodec_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "head" in name:  # newly added parameters
            if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
                new_nodec_params.append(param)
            else:
                new_decay_params.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
                nodec_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "lr_scale": 1.0, "weight_decay": weight_decay},
        {"params": nodec_params, "lr_scale": 1.0, "weight_decay": 0.0},
        {"params": new_decay_params, "lr_scale": lr_scale, "weight_decay": weight_decay},
        {"params": new_nodec_params, "lr_scale": lr_scale, "weight_decay": 0.0},
    ]
    return param_groups


if __name__ == "__main__":
    main()
