#!/bin/bash

DATA_DIR="~/data/VOCdevkit/VOC2012"

echo "Enter output directory:"
read OUPUT_DIR

echo "Enter entropy-based weight (E) or uniform weight (U):"
read WEIGHT

if [ $WEIGHT = "E" ]; then
    echo "=> entropy-based weight"

    python ent_voc_mctformer.py \
    --path MCTformerV2.pth \
    --data_dir $DATA_DIR

    python main_voc_mctformer.py \
    --data_dir $DATA_DIR \
    --batch_size 64 \
    --num_workers 4 \
    --epochs 60 \
    --model deit_small_MCTformerV2_patch16_224 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --cls_epochs 10 \
    --layer_index 7 \
    --score_l 0.35 \
    --score_h 0.55 \
    --use_ent \
    --output_dir $OUTPUT_DIR
else
    echo "=> uniform weight"

    python main_voc_mctformer.py \
    --data_dir $DATA_DIR \
    --batch_size 64 \
    --num_workers 4 \
    --epochs 60 \
    --model deit_small_MCTformerV2_patch16_224 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --cls_epochs 10 \
    --layer_index 7 \
    --score_l 0.35 \
    --score_h 0.55 \
    --output_dir $OUTPUT_DIR
fi

python main_voc_mctformer.py \
--gen_attention_maps \
--split train_aug \
--checkpoint "${OUTPUT_DIR}/checkpoint_best.pth" \
--cam_npy_dir "${OUTPUT_DIR}/cam_npy" \
--cam_png_dir "${OUTPUT_DIR}/cam_png"

python evaluate.py \
--dataset voc \
--split train \
--pred_dir ${OUTPUT_DIR}/cam_npy \
--start 1 --stop 80 \
--type npy \
--write_to ${OUTPUT_DIR}/result.txt
