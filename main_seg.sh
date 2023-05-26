#!/bin/bash

VOC_DIR="~/data/VOCdevkit/VOC2012"
IMG_DIR=${VOC_DIR}/JPEGImages
SEG_DIR=${VOC_DIR}/SegmentationClass
PGT_DIR=""  # directory of pseudo segmentation labels
OUT_DIR=""

python seg/train_seg.py \
    --init_weights "res38_cls.pth" \
    --network resnet38_seg \
    --num_epochs 30 \
    --list_path data/voc/train_aug.txt \
    --img_path ${IMG_DIR} \
    --seg_pgt_path ${PGT_DIR} \
    --save_path  ${OUT_DIR} \
    --num_classes 21 \
    --batch_size 4

python seg/infer_seg.py \
    --weights ${OUT_DIR}/checkpoint.pth \
    --network resnet38_seg \
    --list_path data/voc/val.txt \
    --gt_path ${SEG_DIR} \
    --img_path ${IMG_DIR} \
    --save_path ${OUT_DIR}/val_ms_crf \
    --save_path_c ${OUT_DIR}/val_ms_crf_c \
    --scales 0.5 0.75 1.0 1.25 1.5 \
    --use_crf True
