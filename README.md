# PLCL

## Introduction
Code for paper: Pseudo Labeling based Consistency Learning for Semi-supervised Medical Image Segmentation.

## Requirements
All methods are implemented by Pytorch on a Ubuntu18.04 desktop with NVIDIA RTX2080Ti GPU.

## Usage
1. Place the preprocessed data in '/PLCL/Data'. We will upload the preprocessing code and data in later version.
2. Train the model
```bash
cd code
# eg. for 50 label
python train_PLCL.py --label_num 50 --exp model_name
```
3. Test the model
```bash
cd code
python test_PLCL.py --model model_name  --iter 25000
```
## Ackonwledgements
Our code is origin form [UAMT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MC-net](https://github.com/ycwu1997/MC-Net). Thanks for their valuable works.

## Questions
If you have any question, please contact with us at 'hustapper@gamil.com'.
