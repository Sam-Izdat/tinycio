from __future__ import annotations
import typing
from typing import Union

import torch
from tqdm import tqdm

from ..globals import TINYCIO_VERSION

def version() -> str:
    """Returns tinycio version string"""
    return TINYCIO_VERSION

def version_check_minor(ver_str) -> bool:
    ver = TINYCIO_VERSION.split('.')
    chk = ver_str.split('.')
    return ver[0] == chk[0] and ver[1] == chk[1]

def trilinear_interpolation(im_3d:torch.Tensor, indices:Union[ColorImage, torch.Tensor]) -> torch.Tensor:
    """
    Interpolate 3D image tensor.

    :param im_3d: Input 3D image tensor of shape (C, D, H, W).
    :param indices: Indices into the tensor.
    :return: Interpolated color values.
    """
    # NOTE: Internal - leaving this clutter undocumented intentionally
    indices_floor = indices.floor().to(torch.long)
    indices_ceil = indices.ceil().clamp(0, im_3d.size(0) - 1).to(torch.long)

    weights = (indices - indices_floor).float()

    c000 = im_3d[indices_floor[0],   indices_floor[1],   indices_floor[2]]
    c001 = im_3d[indices_floor[0],   indices_floor[1],   indices_ceil[2]]
    c010 = im_3d[indices_floor[0],   indices_ceil[1],    indices_floor[2]]
    c011 = im_3d[indices_floor[0],   indices_ceil[1],    indices_ceil[2]]
    c100 = im_3d[indices_ceil[0],    indices_floor[1],   indices_floor[2]]
    c101 = im_3d[indices_ceil[0],    indices_floor[1],   indices_ceil[2]]
    c110 = im_3d[indices_ceil[0],    indices_ceil[1],    indices_floor[2]]
    c111 = im_3d[indices_ceil[0],    indices_ceil[1],    indices_ceil[2]]

    interpolated_values = torch.zeros_like(c000).requires_grad_()
    interpolated_values = (
        (1 - weights[0])  * (1 - weights[1])    * (1 - weights[2])    * c000.permute(2,0,1) +
        (1 - weights[0])  * (1 - weights[1])    * weights[2]          * c001.permute(2,0,1) +
        (1 - weights[0])  * weights[1]          * (1 - weights[2])    * c010.permute(2,0,1) +
        (1 - weights[0])  * weights[1]          * weights[2]          * c011.permute(2,0,1) +
        weights[0]        * (1 - weights[1])    * (1 - weights[2])    * c100.permute(2,0,1) +
        weights[0]        * (1 - weights[1])    * weights[2]          * c101.permute(2,0,1) +
        weights[0]        * weights[1]          * (1 - weights[2])    * c110.permute(2,0,1) +
        weights[0]        * weights[1]          * weights[2]          * c111.permute(2,0,1)
    )

    return interpolated_values

class _ProgressBar(tqdm):
    """Provides `update_fit_status(n)` which uses `tqdm.update(delta_n)`."""
    def update_fit_status(self, batches_done=1, steps_per_batch=1, steps_total=None, loss=''):
        if steps_total is not None: self.total = steps_total
        self.set_description('Loss: ' + '{:0.5f}'.format(loss))
        return self.update(batches_done * steps_per_batch - self.n)

def progress_bar(): 
    """Context to display a progressbar with tqdm."""
    return _ProgressBar(unit=' steps', unit_scale=True, unit_divisor=1024, miniters=1, desc="Fitting")