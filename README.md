# Real-World Deep Local Motion Deblurring

[Paper](https://arxiv.org/abs/2204.08179) | [Project Page](https://leiali.github.io/ReLoBlur_homepage/index.html) | [Video](https://youtu.be/mSsADaoh2WY)

## ReLoBlur Dataset
ReLoBlur, the first real-world local motion deblurring dataset, which is captured by a synchronized beam-splitting photographing system. It consists of 2,405 blurred images with the size of 2152×1436 that are divided into 2,010 training images and 395 test images. For efficient training and testing, we also provide the resized version of ReLoBlur Dataset with the size of 538x359. ReLoBlur includes but is not limited to indoor and outdoor scenes of pedestrians, vehicles, parents
and children, pets, balls, plants and furniture.
<img src="assets/ad_data.jpg" width="800px"/>

### Data Download
|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| ReLoBlur test | [link]() | [link](https://drive.google.com/drive/folders/1nYj4e7TSXeqBsUZxLvoay_JLZ7wxdNmC?usp=sharing) | 395 | We provide 395 pairs of testing images.|
| ReLoBlur train| [link]() | [link](https://drive.google.com/drive/folders/1rAPKzhhRjztj7Utbb00BJLSVaPC-1Jua?usp=sharing) | 2,010 | We provide 2010 pairs of training images.|
| Resized | [link]() | [link](https://drive.google.com/drive/folders/1M_5O-fGqvCry1AmY0JhE2DulbZguIeA3?usp=sharing) | 2,405 | We provide a resized version of ReLoBlur dataset. We resized ReLoBlur by BICUBIC algorithm. |
| Local Blur Mask | [link]() | [link](https://drive.google.com/drive/folders/1-4YerKKlDydgoBeZbiV0_XR9iJLKbLXI?usp=sharing) | 2,405 | We provide a resized version of ReLoBlur dataset. We resized ReLoBlur by BICUBIC algorithm. |

### ReLoBlur structure
```
├── ReLoBlur
    ├── train
         ├── s01
             ├── 00
             ├── 01
             ├── ...
         ├── s02
             ├── 00
             ├── ...
         ├── ...      
    ├── test
         ├── scene1
              ├── 00
              ├── 01
              ├── ...
         ├── scene2
              ├── 00
              ├── 01
              ├── ...
``

Important: ReLoBlur dataset can be only used for academic purposes.

## LBAG: Local Blur-Aware Gated Network
Based on ReLoBlur, we propose a Local Blur-Aware Gated network (LBAG) and several local blur-aware techniques to bridge the gap between global and local deblurring. LBAG detects blurred regions and restores locally blurred images simultaneously. 

### Environment

Before running LBAG, please install the environment on Linux:

```
pip install -r requirements.txt
```

### Pre-processing

In order to fasten the data loading speed, we firstly binarize data by running:

```
python data_gen/binarize_relo.py
```

### Pretrained Model

LBAG+ uses the pretrained model of MiMO_UNet. The pretrained models could be downloaded here: https://github.com/chosj95/MIMO-UNet

And please put the pretrained model into 
```
checkpoints/pretrained_mimounet
```

### Evaluation Code

You can evaluate the LBAG by using:
```
CUDA_VISIBLE_DEVICES=0 python LBAG_trainer.py --config configs/LBAG.yaml --exp_name 0223_munet_gate01_1 --reset --infer
```

### Training

**Training with single GPU**

To train a model with your own data/model, you can edit the `configs/LBAG.yaml` and run the following codes:

```
CUDA_VISIBLE_DEVICES=0 python LBAG_trainer.py --config configs/LBAG.yaml --exp_name 0223_munet_gate01_1 --reset
```

### Citation

Coming soon...
