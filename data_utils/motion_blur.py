import kornia
import torch
import numpy as np
from cv2 import cv2
from scipy.special import binom

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1;
        self.p2 = p2
        self.angle1 = angle1;
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


def get_random_shape_mask(h, w, x1, x2, y1, y2):
    rad = 0.2
    edgy = 0.05
    a = get_random_points(n=7, scale=1)
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    mask = np.zeros((h, w))  # create a single channel 200x200 pixel black image
    contours = np.around(np.array([[x2 - x1, y2 - y1]]) * np.array(list(zip(x, y)))).astype(int)
    contours = contours + np.array([x1, y1])
    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
    return mask


def motion_deblur(img, mask=1, ks=5, ang=60, direction=0):
    output = kornia.filters.motion_blur(
        torch.FloatTensor(img).permute(2, 0, 1)[None, ...],
        ks, ang, direction, mode='bilinear').numpy()[0].transpose(1, 2, 0)
    output = np.round(output).astype(np.uint8)
    output = output * mask + img * (1 - mask)
    return output


if __name__ == '__main__':
    from data_utils.meters import Timer
    from data_utils.detr2_utils import get_segmentation_masks, build_predictor

    img_path = '/workspace/lihaoying/BS_local_deblur/LBAG/data/test/20211214_20%/f04_bst/f04_20211214144307-10_cam1_sharp.bmp'
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    size = 5
    predictor = build_predictor()
    seg_masks = get_segmentation_masks(img, predictor)
    seg_masks = seg_masks.numpy()
    for seg_mask in seg_masks[:1]:
        seg_mask = seg_mask[:, :, None]
        with Timer(enable=True, name='motionblur'):
            output = motion_deblur(img, seg_mask, direction=0.5, ks=11, ang=0)
            # cv2.imshow('Motion Blur', output / 255)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
