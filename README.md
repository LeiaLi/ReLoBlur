# Real-World Deep Local Motion Deblurring

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25215) | [Project Page](https://leiali.github.io/ReLoBlur_homepage/index.html) | [Video](https://youtu.be/mSsADaoh2WY)

## ReLoBlur Dataset

ReLoBlur, the first real-world local motion deblurring dataset, which is captured by a synchronized beam-splitting photographing system. It consists of 2,405 blurred images with the size of 2,152×1,436 that are divided into 2,010 training images and 395 test images. For efficient training and testing, we also provide the resized version of ReLoBlur Dataset with the size of 538x359. ReLoBlur includes but is not limited to indoor and outdoor scenes of pedestrians, vehicles, parents
and children, pets, balls, plants and furniture.
<img src="assets/ad_data.jpg" width="800px"/>

### Data Download
|     | Google Drive | Baidu Cloud | Number | Description|
| :--- | :----: |:----: |:---- | ---- |
| ReLoBlur test | [link](https://drive.google.com/drive/folders/1nYj4e7TSXeqBsUZxLvoay_JLZ7wxdNmC?usp=sharing) | [link](https://pan.baidu.com/s/1dBIs95-KlFTth9cqjX23-Q?pwd=nmcy) (code:nmcy)| 395 | We provide 395 pairs of testing images.|
| ReLoBlur train| [link](https://drive.google.com/drive/folders/1rAPKzhhRjztj7Utbb00BJLSVaPC-1Jua?usp=sharing) | [link](https://pan.baidu.com/s/1CoHScOL46_L06LGXg3K2lg?pwd=49nb) (code:49nb) | 2,010 | We provide 2,010 pairs of training images.|
| Local Blur Mask | [link](https://drive.google.com/drive/folders/1-4YerKKlDydgoBeZbiV0_XR9iJLKbLXI?usp=sharing) | [link](https://pan.baidu.com/s/1p6Z_EJhjVvRxWu92VYqe9Q?pwd=98mw) (code:98mw) | 2,405 | We provide a resized version of ReLoBlur dataset. We resized ReLoBlur by the BICUBIC algorithm. |

The resized version of the ReLoBlur dataset can be processed by BICUBIC algorithm.

Important: ReLoBlur dataset can be only used for academic purposes!

## News
- **2024.10**: Update the paper link.
- **2024.4**: Paper [LMD-ViT](https://github.com/LeiaLi/LMD-ViT) releases [a new version of annotated blur masks](https://drive.google.com/drive/folders/1cBhtfm7vzsyAr9D6V_LwWJma845rUSlg?usp=drive_link).
- **2023.3**: Code, model, and dataset are released.
- **2022.11**: Paper "Real-world Deep Local Motion Deblurring" is accepted.

## LBAG: Local Blur-Aware Gated Network
Based on ReLoBlur, we propose a Local Blur-Aware Gated network (LBAG) and several local blur-aware techniques to bridge the gap between global and local deblurring. LBAG detects blurred regions and restores locally blurred images simultaneously. 

### Environment

Before running LBAG, please install the environment on Linux:

```
pip install -U pip
pip install -r requirements.txt
```

### Pre-processing

Put the training data and testing data under
```
LBAG/data/dataset/
```
and put the masks under
```
LBAG/data/
```

If you train or infer with your own data, the data structure should be like:
```
├── dataset
    ├── train
         ├── s01
             ├── 00
                ├── 00_sharp.png
                ├── 00_blur.png
                ├── ...
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
├── masks
    ├── train
         ├── s01
             ├── 00
                 ├── 00.png
                 ├── 01.png
                 ├── ...
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
```

### Pretrained Model

LBAG+ uses the pretrained model of MiMO_UNet, which could be downloaded [here](https://github.com/chosj95/MIMO-UNet)

Please put the pre-trained model into 
```
checkpoints/pretrained_mimounet/
```
You can infer LBAG by the pre-trained model, which could be downloaded [here](https://drive.google.com/drive/folders/1_T8Z7-T7i7BsaJhpq0rCNWLT65N_SslU?usp=sharing)

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

## Cite

If the dataset or code helps your research or work, please cite our paper or star this repo. Thank you!
```
@inproceedings{li2023real,
  title={Real-world deep local motion deblurring},
  author={Li, Haoying and Zhang, Ziran and Jiang, Tingting and Luo, Peng and Feng, Huajun and Xu, Zhihai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={1314--1322},
  year={2023}
}
```
