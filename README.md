# UDAMT: Uncertainty-Driven Auxiliary Mean Teacher for Skin Lesion Segmentation

## Requirements
- Python 3.8
- PyTorch 1.8.1
- CUDA 11.1
- Install dependencies: `pip install -r requirements.txt`

## Dataset Preparation
1. Download ISIC 2016/2017/2018 datasets from [ISIC Archive](https://challenge.isic-archive.com/data).
2. Place images in `data/ISIC2018/images` and masks in `data/ISIC2018/masks`.

## Training
```bash
python train.py --config configs/udamt.yaml --labeled_ratio 0.05