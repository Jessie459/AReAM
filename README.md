# AReAM

PyTorch code for [Mitigating Undisciplined Over-Smoothing in Transformer for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2305.03112).

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>
<p align="center">
Overview of AReAM.
</p>

## PASCAL VOC 2012

### Data Preparation

Download [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html).

``` bash
VOCdevkit/
└── VOC2012
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    ├──...
```

### Training

Step 1: train the classification model, generate and evaluate CAMs

```
# Option 1: MCTformer backbone
bash main_voc_mctformer.sh

# Option 2: Conformer backbone
bash main_voc_conformer.sh
```

| Backbone | Weight | Checkpoint | mIoU (%) |
| :------: | :-----: | :-------: | :------: |
| MCTformer | E | [Link](https://drive.google.com/file/d/1pYAZPdXZGrjmZeUJq59x-C_9bNYbdDiD/view?usp=share_link) | 67.73 |
| MCTformer | U | [Link](https://drive.google.com/file/d/1iEI3iej29wFLfT7gPXBU2x5BneeYklpV/view?usp=sharing) | 67.93 |
| Conformer | E | [Link]() | |
| Conformer | U | [Link]() | |

Step 2: Run the run_psa.sh script for using [PSA](https://github.com/jiwoon-ahn/psa) to post-process the seeds (i.e., class-specific localization maps) to generate pseudo ground-truth segmentation masks. To train PSA, the pre-trained classification [weights](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing) were used for initialization.
```
bash run_psa.sh
```

Step 3: For the segmentation part, run the run_seg.sh script for training and testing the segmentation model. When training on VOC, the model was initialized with the pre-trained classification [weights](https://drive.google.com/file/d/1xESB7017zlZHqxEWuh1Rb89UhjTGIKOA/view?usp=sharing) on VOC.
```
bash run_seg.sh
```