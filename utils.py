import subprocess

import numpy as np
import torch
import torch.distributed as dist
import glob
import os
import re


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def tensors_to_np(tensors):
    if isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = tensors_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = tensors_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))


def load_checkpoint(model, optimizer, work_dir):
    checkpoint, _ = get_last_checkpoint(work_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.cuda()
    return training_step


def save_checkpoint(model, optimizer, work_dir, global_step, num_ckpt_keep):
    ckpt_path = f'{work_dir}/model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]:
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def plot_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)
