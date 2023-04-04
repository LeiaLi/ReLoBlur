import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from criterions.ssim_loss import SSIM
from data_utils.align import image_align
from dataset import DeblurDataSet
from hparams import hparams, set_hparams
from models.MIMOUNet import LBAG
import numpy as np
import torch.nn.functional as F

from utils import plot_img, move_to_cuda, load_checkpoint, save_checkpoint, tensors_to_scalars


class LBAGTrainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.metric_keys = ['psnr', 'ssim', 'weighted_psnr', 'weighted_ssim', 'aligned_psnr']
        self.ssim_loss = SSIM(window_size=11, size_average=False)
        self.work_dir = hparams['work_dir']
        self.first_val = True

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_model(self):
        self.model = LBAG(num_res=20)
        if hparams['pretrained_ckpt'] != '':
            state = torch.load(hparams['pretrained_ckpt'], map_location='cpu')
            del state['model']['feat_extract.5.main.0.weight']
            del state['model']['feat_extract.5.main.0.bias']
            del state['model']['ConvsOut.0.main.0.weight']
            del state['model']['ConvsOut.0.main.0.bias']
            del state['model']['ConvsOut.1.main.0.weight']
            del state['model']['ConvsOut.1.main.0.bias']
            self.model.load_state_dict(state['model'], strict=False)
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=0)

    def build_scheduler(self, optimizer):
        lr_steps = [(x + 1) * 100000 for x in range(600000 // 100000)]
        gamma = 0.5
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, gamma)

    def build_train_dataloader(self):
        dataset = DeblurDataSet('train')
        return torch.utils.data.DataLoader(
            dataset, batch_size=hparams['batch_size'], shuffle=True,
            pin_memory=False, num_workers=hparams['num_workers'])

    def build_val_dataloader(self):
        return torch.utils.data.DataLoader(
            DeblurDataSet('test'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def train(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
        scheduler = self.build_scheduler(optimizer)
        scheduler.step(training_step)
        dataloader = self.build_train_dataloader()

        train_pbar = tqdm(dataloader, initial=training_step, total=float('inf'),
                          dynamic_ncols=True, unit='step')
        for batch in train_pbar:
            if training_step % hparams['val_check_interval'] == 0:
                with torch.no_grad():
                    model.eval()
                    self.validate(training_step)
                save_checkpoint(model, optimizer, self.work_dir, training_step, hparams['num_ckpt_keep'])
            model.train()
            batch = move_to_cuda(batch)

            optimizer.zero_grad()
            img_out, gate_xs = model(batch['img_blur'])
            img_g = batch['img_gt']
            losses = {}
            imgs_g = [
                F.interpolate(img_g, scale_factor=0.25, mode='bilinear'),
                F.interpolate(img_g, scale_factor=0.5, mode='bilinear'),
                img_g,
            ]
            blur_mask = batch['blur_mask']
            blur_masks = [
                F.interpolate(blur_mask, scale_factor=0.25, mode='nearest'),
                F.interpolate(blur_mask, scale_factor=0.5, mode='nearest'),
                blur_mask,
            ]
            losses_, total_loss = self.calc_losses(img_out, imgs_g)
            losses.update(losses_)
            if not hparams['multiscale_gate']:
                blur_masks = blur_masks[-1:]
            for i_s, (gate_x, bm) in enumerate(zip(gate_xs, blur_masks)):
                losses[f'g_{i_s}'] = F.mse_loss(gate_x, bm)
                total_loss += losses[f'g_{i_s}'] * hparams['lambda_gate']
            total_loss.backward()
            optimizer.step()
            training_step += 1
            scheduler.step(training_step)
            if training_step % 100 == 0:
                self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
            train_pbar.set_postfix(**tensors_to_scalars(losses))

    def calc_losses(self, imgs_p, imgs_g, blur_masks=None, suffix=''):
        losses = {}
        total_loss = 0
        if not hparams['multiscale']:
            imgs_p = imgs_p[-1:]
            imgs_g = imgs_g[-1:]
            if blur_masks is not None:
                blur_masks = blur_masks[-1:]
        for i_img, (img_p, img_g) in enumerate(zip(imgs_p, imgs_g)):
            B, N_rpt = img_p.shape[0], 1
            blur_mask = blur_masks[i_img] if blur_masks is not None else None
            if hparams['shift_loss']:
                shifted_dxys = hparams['shifted_dxys']
                N_rpt = len(shifted_dxys) ** 2
                img_shift_g = []
                for dy in shifted_dxys:
                    for dx in shifted_dxys:
                        img_shift_g.append(F.pad(img_g, [dx, -dx, dy, -dy], mode='reflect'))
                img_g = torch.stack(img_shift_g, 1)
                img_g = img_g.flatten(0, 1)
                img_p = img_p[:, None, ...].repeat([1, N_rpt, 1, 1, 1])
                img_p = img_p.flatten(0, 1)
                if blur_mask is not None:
                    blur_mask = blur_mask[:, None, ...].repeat([1, N_rpt, 1, 1, 1])
                    blur_mask = blur_mask.flatten(0, 1)
            metric_keys = [k for k in ['l1', 'ssim', 'fft', 'fft_amp'] if hparams[f'lambda_{k}'] > 0]
            losses_ = {k: getattr(self, f'{k}_loss')(img_p, img_g, blur_mask).reshape(B, N_rpt).amin(1)
                       for k in metric_keys}
            losses_ = {k: v.mean() for k, v in losses_.items() if not torch.isnan(v).any()}
            total_loss += sum([v * hparams[f'lambda_{k}'] for k, v in losses_.items()])
            for k, v in losses_.items():
                losses[f'{k}S{i_img}'] = v.item()
        losses = {f'{k}{suffix}': v for k, v in losses.items()}
        return losses, total_loss

    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        results = {k: [] for k in self.metric_keys}
        for batch_idx, batch in pbar:
            if self.first_val and batch_idx > hparams['num_sanity_val_steps']:  # 每次运行的第一次validation只跑一小部分数据，来验证代码能否跑通
                break
            batch = move_to_cuda(batch)
            img_out, gate_xs = self.model(batch['img_blur'])
            self.gate_xs = gate_xs
            img_out = img_out[-1]
            if batch_idx in hparams['valid_plot_samples'] and img_out is not None and \
                    training_step % hparams['valid_plot_interval'] == 0 and \
                    not hparams['infer']:
                img_name = f'{batch_idx}_{batch["item_name"][0].replace("/", "_")}'
                if img_out is not None:
                    self.logger.add_image(f'{img_name}_2p', plot_img(img_out[0]), training_step)
                    if hparams['model_type'] == 2:
                        self.logger.add_image(f'{img_name}_5gate', plot_img(self.gate_xs[-1][0]), training_step)
                if training_step <= hparams['val_check_interval']:
                    img_noisy = batch['img_blur']
                    img_gt = batch['img_gt']
                    blur_mask = batch['blur_mask']
                    self.logger.add_image(f'{img_name}_1g', plot_img(img_gt[0]), training_step)
                    self.logger.add_image(f'{img_name}_3b', plot_img(img_noisy[0]), training_step)
                    self.logger.add_image(f'{img_name}_4m', plot_img(blur_mask[0]), training_step)
            results = self.calc_metrics(img_out, batch['img_gt'], results, weights=batch.get('blur_mask_nonpad'))
            metrics = {k: np.mean(results[k]) for k in self.metric_keys}
            # ret_ = self.calc_metrics(batch['img_blur'], batch['img_gt'], weights=batch.get('blur_mask_nonpad'))
            # metrics.update({k + '_min': np.mean(ret_[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if hparams['infer']:
            print('Val results:', metrics)
        else:
            if not self.first_val:
                self.log_metrics({f'val/{k}': v for k, v in metrics.items()}, training_step)
                print('Val results:', metrics)
            else:
                print('Sanity val results:', metrics)
        self.first_val = False

    def test(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
        with torch.no_grad():
            model.eval()
            self.first_val = False
            self.validate(training_step)

    ###################
    # losses and metrics
    ###################
    def fft_loss(self, img_p, img_g, weight=None):
        pred_fft = torch.fft.rfft2(img_p)
        label_fft = torch.fft.rfft2(img_g)
        return F.l1_loss(pred_fft, label_fft, reduction='none').mean([1, 2, 3])

    def l1_loss(self, img_p, img_g, weight=None):
        l1_loss = F.l1_loss(img_p, img_g, reduction='none')
        if weight is None:
            return l1_loss.mean([1, 2, 3])
        return (l1_loss * weight).sum([1, 2, 3]) / weight.sum([1, 2, 3]).clamp_min(1) / 3

    def psnr_metric(self, img_p, img_g):
        w_psnr1 = ((img_p - img_g) ** 2).mean()
        w_psnr = 10 * torch.log10(1 / w_psnr1)
        return w_psnr

    def weighted_psnr_metric(self, img_p, img_g, weight):
        w_psnr1 = (((img_p - img_g) ** 2) * weight).sum() / weight.sum() / 3
        w_psnr = 10 * torch.log10(1 / w_psnr1)
        return w_psnr

    def aligned_psnr_metric(self, img_p, img_g):
        img_p = img_p.permute(1, 2, 0)
        img_g = img_g.permute(1, 2, 0)

        aligned_deblurred, aligned_xr1, cr1, shift = image_align(img_p.cpu().numpy(), img_g.cpu().numpy())
        w_psnr = torch.FloatTensor(((aligned_deblurred - aligned_xr1) ** 2)).mean()
        w_psnr = 10 * torch.log10(1 / w_psnr)
        return w_psnr

    def ssim_metric(self, img_p, img_g):
        ssim = -self.ssim_loss(img_p[None, ...], img_g[None, ...])[0]
        return ssim

    def weighted_ssim_metric(self, img_p, img_g, weight):
        ssim = -self.ssim_loss(img_p[None, ...], img_g[None, ...], weight[None, ...])[0]
        return ssim

    def calc_metrics(self, img_p, img_g, results=None, weights=None):
        if results is None:
            results = {k: [] for k in self.metric_keys}
        for i in range(img_g.shape[0]):
            if weights is not None:
                weight = weights[i]
            img_p = img_p[i]
            img_p = torch.clamp(img_p, 0, 1)
            img_p = torch.round(img_p * 255).long().float() / 255
            img_g = img_g[i]
            img_g = torch.clamp(img_g, 0, 1)
            img_g = torch.round(img_g * 255).long().float() / 255

            for k in self.metric_keys:
                if 'weighted_' in k:
                    if weights is not None:
                        results[k].append(getattr(self, f'{k}_metric')(img_p, img_g, weight).cpu().numpy())
                else:
                    results[k].append(getattr(self, f'{k}_metric')(img_p, img_g).cpu().numpy())
        return results

    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics


if __name__ == '__main__':
    set_hparams()
    trainer = LBAGTrainer()
    if not hparams['infer']:
        trainer.train()
    else:
        trainer.test()
