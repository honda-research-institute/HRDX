"""
Distributed testing utilities compatible with MMEngine runners.
"""

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import shutil
import tempfile
import time

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
from mmengine.dist import get_dist_info
from mmengine.fileio import dump, load
from mmengine.utils import ProgressBar, mkdir_or_exist


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code (semantic masks only)."""
    cls_segms = mask_results
    encoded_mask_results = []
    for segm in cls_segms:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(segm[:, :, np.newaxis], order='F', dtype='uint8'))[0])
    return [encoded_mask_results]


def _unwrap_data_container(value, key=None):
    """Convert vendored DataContainer objects back to plain tensors/lists."""
    if hasattr(value, 'data'):
        data = value.data
        if isinstance(data, (list, tuple)) and data:
            if key == 'seq_info':
                return data
            return data[0]
        return data
    return value


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple GPUs and aggregate bbox/mask results."""
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # Avoid deadlocks observed on some clusters.
    have_mask = False

    for data in data_loader:
        with torch.no_grad():
            clean_data = {k: _unwrap_data_container(v, k) for k, v in data.items()}
            result = model(return_loss=False, rescale=True, **clean_data)

        if isinstance(result, dict):
            if 'bbox_results' in result:
                bbox_result = result['bbox_results']
                batch_size = len(bbox_result)
                bbox_results.extend(bbox_result)
            else:
                batch_size = len(result)
            if 'mask_results' in result and result['mask_results'] is not None:
                mask_result = custom_encode_mask_results(result['mask_results'])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(result)
            bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        mask_results = collect_results_gpu(mask_results, len(dataset)) if have_mask else None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        mask_tmpdir = f'{tmpdir}_mask' if tmpdir is not None else None
        mask_results = (collect_results_cpu(mask_results, len(dataset), mask_tmpdir)
                        if have_mask else None)

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results on CPU by saving them to temporary files."""
    rank, world_size = get_dist_info()
    if tmpdir is None:
        max_len = 512
        dir_tensor = torch.full((max_len,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir_tensor = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir_tensor)] = tmpdir_tensor
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(load(part_file))

    ordered_results = []
    for res in part_list:
        ordered_results.extend(list(res))

    print(f'\ntruncate {size} samples from {len(ordered_results)}')
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results on GPU; fall back to the CPU implementation."""
    return collect_results_cpu(result_part, size)
