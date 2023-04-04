import glob
import subprocess
import os
import numpy as np
import cv2
import pdb

mask_dir_list = ['/data/hyli/LBAG/data/processed/test2', '/data/hyli/LBAG/data/processed/train']
mask_area = 0
image_area = 0
for mask_dir in mask_dir_list:
    sub_dir_list = sorted(glob.glob(f'{mask_dir}/*'))
    for sub_dir in sub_dir_list:
        sub_sub_dir_list = sorted(glob.glob(f'{sub_dir}/*'))
        for sub_sub_dir in sub_sub_dir_list:
            mask_image_list = sorted(glob.glob(f'{sub_sub_dir}/*.jpg'))
            for mask in mask_image_list:
                print(mask)
                image = cv2.imread(mask)
                first_layer = image[:,:,0]
                cnt_array = np.count_nonzero(first_layer)
                # mask_area = len(first_layer.nonzero(first_layer)[0])
                mask_area = mask_area + np.sum(cnt_array)
                # pdb.set_trace()
                image_area = image_area+first_layer.shape[0] * first_layer.shape[1] 
                print(mask_area)
                # pdb.set_trace()
ratio = mask_area/image_area
print(ratio)