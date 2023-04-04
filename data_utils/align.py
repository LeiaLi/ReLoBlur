import cv2
import pdb
import numpy as np


def image_align(deblurred, gt):
    # this function is based on kohler evaluation code
    z = deblurred
    c = np.ones_like(z)
    x = gt

    zs = (np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    termination_eps = 0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
                                             warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

    target_shape = x.shape
    shift = warp_matrix

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT)

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, shift
