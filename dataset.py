import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data_utils import PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCompose
from torchvision.transforms import functional as tf
import torch.nn.functional as F
import pdb

from hparams import hparams


class DeblurDataSet(Dataset):
    def __init__(self, prefix='train'):
        self.hparams = hparams
        self.data_dir = hparams['data_dir'] #
        self.prefix = prefix
        if prefix == 'train':
            self.transform = PairCompose(
                [
                    PairRandomCrop(hparams['patch_size']),
                    PairRandomHorizontalFilp(),
                    PairToTensor()
                ]
            )
            self.transform_nocrop = PairCompose(
                [
                    PairRandomHorizontalFilp(),
                    PairToTensor()
                ]
            )
        else:
            self.transform = None
        # self.npy_filenames = sorted(glob.glob(f'{self.data_dir}/{self.prefix}/*.npy'))
        self.sharp_filenames = sorted(glob.glob(f'{self.data_dir}/{self.prefix}/*/*/*sharp.png'))
        self.l = len(self.sharp_filenames)

    def __getitem__(self, index):
        gt_path = self.sharp_filenames[index]
        blur_path = gt_path.replace('sharp','blur')
        mask_path = gt_path.replace('dataset','masks')
        mask_path = mask_path.replace('_sharp', '')
        item_path = "/".join(gt_path.split("/")[3:])
        item_path = item_path.split('_sharp')[0]
        
        img_gt = Image.open(gt_path) 
        img_gt = np.uint8(np.array(img_gt))
        img_blur = Image.open(blur_path)
        img_blur = np.uint8(np.array(img_blur))
        blur_mask = Image.open(mask_path)
        blur_mask = np.uint8(np.array(blur_mask))
        if len(blur_mask.shape) == 2:
            blur_mask = blur_mask[..., None].repeat(repeats=3, axis=2)
        # blur_mask = np.tile(np.uint8(item['blur_mask'].toarray() * 255)[:, :, None], [1, 1, 3])
        if self.prefix == 'train':
            force_blur_region = \
                self.hparams.get('force_blur_region_p', 0.0) > 0 and \
                random.random() < self.hparams['force_blur_region_p']
            ps = self.hparams['patch_size']
            if ps > 0:
                H, W, _ = img_gt.shape
                y_s, x_s = random.randint(0, H - 1 - ps), random.randint(0, W - 1 - ps)
                R_blur = blur_mask.sum() / np.prod(blur_mask.shape)
                if force_blur_region and R_blur > 0.05:
                    blur_pixs = np.stack(np.where(blur_mask[ps // 2:-ps // 2, ps // 2:-ps // 2, 0]), 1)
                    if len(blur_pixs) > 0:
                        y_s, x_s = random.choice(blur_pixs)

                img_gt = img_gt[y_s:y_s + ps, x_s:x_s + ps]
                img_blur = img_blur[y_s:y_s + ps, x_s:x_s + ps]
                blur_mask = blur_mask[y_s:y_s + ps, x_s:x_s + ps]

            img_gt = Image.fromarray(img_gt.astype(np.uint8)).convert('RGB')
            img_blur = Image.fromarray(img_blur.astype(np.uint8)).convert('RGB')
            blur_mask = Image.fromarray(blur_mask.astype(np.uint8))
            img_gt, img_blur, blur_mask = self.transform_nocrop([img_gt, img_blur, blur_mask])
        else:
            img_gt, img_blur, blur_mask = tf.to_tensor(img_gt), tf.to_tensor(img_blur), tf.to_tensor(blur_mask)
            multiple_width = self.hparams['multiple_width']
            if img_gt.shape[1] % multiple_width != 0:
                l_pad = multiple_width - img_gt.shape[1] % multiple_width
                img_blur = F.pad(img_blur[None, ...], [0, 0, 0, l_pad], mode='reflect')[0]
                blur_mask = F.pad(blur_mask[None, ...], [0, 0, 0, l_pad], mode='reflect')[0]
            if img_gt.shape[2] % multiple_width != 0:
                l_pad = multiple_width - img_gt.shape[2] % multiple_width
                img_blur = F.pad(img_blur[None, ...], [0, l_pad, 0, 0], mode='reflect')[0]
                blur_mask = F.pad(blur_mask[None, ...], [0, l_pad, 0, 0], mode='reflect')[0]
        sample = {'img_gt': img_gt, 'img_blur': img_blur, 'item_name': item_path}
        blur_mask = (blur_mask > 0).float()[:1]  # [B, 1, H, W]
        blur_mask_nonpad = blur_mask[:, :1436, :2152]
        sample['blur_mask'] = blur_mask
        sample['blur_mask_nonpad'] = blur_mask_nonpad
        return sample

    def __len__(self):
        return self.l
