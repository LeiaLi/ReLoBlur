import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=8, ave_spectrum=False,
                 log_matrix=False, batch_matrix=False, weighted=False):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.weighted = weighted

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        win_h = patch_h
        win_w = patch_w
        for i in range(patch_factor):
            for j in range(patch_factor):
                if (i + 1) * patch_h + win_h <= h and (j + 1) * patch_w + win_w <= w:
                    patch_list.append(x[:, :,
                                      i * patch_h:(i + 1) * patch_h + win_h,
                                      j * patch_w:(j + 1) * patch_w + win_w])
        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        freq = torch.fft.fft2(y)
        freq_amp = (freq.real ** 2 + freq.imag ** 2 + 1e-8).sqrt()
        return freq_amp

    def pooling_weight(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        win_h = patch_h
        win_w = patch_w
        for i in range(patch_factor):
            for j in range(patch_factor):
                if (i + 1) * patch_h + win_h <= h and (j + 1) * patch_w + win_w <= w:
                    patch_list.append(x[:,
                                      i * patch_h:(i + 1) * patch_h + win_h,
                                      j * patch_w:(j + 1) * patch_w + win_w])
        # stack to patch tensor
        y = torch.stack(patch_list, 1)
        y = y.amax([2, 3])
        return y

    def forward(self, pred, target):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq_amp = self.tensor2freq(pred)
        target_freq_amp = self.tensor2freq(target)
        l1 = F.l1_loss(pred_freq_amp, target_freq_amp, reduction='none')
        return l1.mean([1, 2, 3, 4])
