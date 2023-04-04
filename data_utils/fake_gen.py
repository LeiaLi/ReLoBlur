from data_utils.motion_blur import motion_deblur
from data_utils.meters import Timer
from data_utils.detr2_utils import get_segmentation_masks, build_predictor
import cv2
import glob
from subprocess import call
from tqdm import tqdm
import pdb

if __name__ == '__main__':

    # img_path = '/workspace/lihaoying/BS_local_deblur/LBAG/data/test/20211214_20%/f04_bst/f04_20211214144307-10_cam1_sharp.bmp'
    main_dir = 'data/test/20211214_20%/f05_bst/'
    img_path = sorted(glob.glob(f'{main_dir}*_sharp.bmp'))
    sub_dir = "/".join(main_dir.split("/")[-3:])
    # print(sub_dir)
    # pdb.set_trace()
    save_path = f'data/synthetic_blur/'
    if not call(f'find {save_path} -name "{sub_dir}"', shell=True):  # 如果没有该文件夹，则建立
        call(f'mkdir {save_path}{sub_dir}', shell=True)

    for img_pth in tqdm(list(img_path)):
        img_name = img_pth.split('/')[-1]
        img_name = "_".join(img_name.split('_')[:2])
        print(img_name)
        # pdb.set_trace()
        img = cv2.imread(img_pth)
        H, W, _ = img.shape
        size = 5
        predictor = build_predictor()
        seg_masks = get_segmentation_masks(img, predictor)
        seg_masks = seg_masks.numpy()
        for seg_mask in seg_masks[:1]:
            seg_mask = seg_mask[:, :, None]
            with Timer(enable=True, name='motionblur'):
                output = motion_deblur(img, seg_mask, direction=0.5, ks=11, ang=0)
                cv2.imwrite(f'{save_path}{sub_dir}/syn_{img_name}_blur.bmp', output)