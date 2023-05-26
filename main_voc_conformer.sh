#!/bin/bash

DATA_DIR="~/data/VOCdevkit/VOC2012"

echo "Enter output directory:"
read OUTPUT_DIR

echo "Enter entropy-based weight (E) or uniform weight (U):"
read WEIGHT

if [ $WEIGHT = "E" ]; then
    echo "=> entropy-based weight"

    python main_voc_conformer.py \
    --pretrained pretrained/conformer_sm.pth \
    --data_dir $DATA_DIR \
    --batch_size 8 \
    --num_workers 2 \
    --epochs 12 \
    --cls_epochs 2 \
    --layer_index 9 \
    --score_l 0.35 \
    --score_h 0.55 \
    --use_ent \
    --output_dir $OUTPUT_DIR
else
    echo "=> uniform weight"

    python main_voc_conformer.py \
    --pretrained pretrained/conformer_sm.pth \
    --data_dir $DATA_DIR \
    --batch_size 8 \
    --num_workers 2 \
    --epochs 12 \
    --cls_epochs 2 \
    --layer_index 9 \
    --score_l 0.35 \
    --score_h 0.55 \
    --output_dir $OUTPUT_DIR
fi

python infer_conformer.py \
--dataset voc \
--split train_aug \
--checkpoint "${OUTPUT_DIR}/checkpoint.pth" \
--cam_npy_dir "${OUTPUT_DIR}/cam_npy" \
--cam_png_dir "${OUTPUT_DIR}/cam_png" \
--scales 0.5 1.0 1.5

python evaluate.py \
--dataset voc \
--split train \
--pred_dir ${OUTPUT_DIR}/cam_npy \
--start 1 --stop 80 \
--type npy \
--write_to ${OUTPUT_DIR}/result.txt
