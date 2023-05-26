#!/bin/bash

VOC_DIR=""
IMG_DIR=${VOC_DIR}/JPEGImages
CAM_DIR=""
OUT_DIR=""

# refine with dense CRF
# python infer_crf.py --dataset voc --img_dir ${IMG_DIR} --cam_dir ${CAM_DIR} --out_dir ${OUT_DIR} --score 0.25 0.60 --visualize
python infer_crf.py --dataset voc --img_dir ${IMG_DIR} --cam_dir ${CAM_DIR} --out_dir ${OUT_DIR} --alpha 1 12 --visualize

# train PSA
# python psa/train_aff.py \
#     --voc12_root ${VOC_DIR} \
#     --la_crf_dir ${OUT_DIR}/crf_score0.60 \
#     --ha_crf_dir ${OUT_DIR}/crf_score0.25 \
#     --weights res38_cls.pth

python psa/train_aff.py \
    --weights res38_cls.pth \
    --voc12_root ${VOC_DIR} \
    --la_crf_dir ${OUT_DIR}/crf_alpha1 \
    --ha_crf_dir ${OUT_DIR}/crf_alpha12

# infer PSA
python psa/infer_aff.py \
    --weights resnet38_aff.pth \
    --infer_list data/voc/train_aug.txt \
    --cam_dir ${CAM_DIR} \
    --voc12_root ${VOC_DIR} \
    --out_rw ${OUT_DIR}/psa

# evaluate pseudo segmentation labels
python evaluate.py \
    --dataset voc \
    --split train \
    --pred_dir ${OUT_DIR}/psa \
    --type png \
    --write_to ${OUT_DIR}/psa_result.txt
