from cv2 import cv2
import numpy as np
from scipy import signal
# import numba as nb
#
kernelX = cv2.getGaussianKernel(11, 1.5)


# @nb.jit()
def cal_ssim(img1, img2, weight=None):
    K = [0.01, 0.03]
    L = 255
    window = kernelX * kernelX.T

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if weight is None:
        mssim = np.mean(ssim_map)
    else:
        identity_window = np.zeros_like(window)
        identity_window[5, 5] = 1
        weight = signal.convolve2d(weight[0], identity_window, 'valid')
        mssim = (ssim_map * weight).sum() / weight.sum()
    return mssim, ssim_map
