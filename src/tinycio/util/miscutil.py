from __future__ import annotations
import typing
from typing import Union

import torch
import numpy as np
from tqdm import tqdm

from ..globals import TINYCIO_VERSION

def version() -> str:
    """
    Get current tinycio version.
    :return: version string ("major.minor.patch")
    """
    return TINYCIO_VERSION

def version_check_minor(ver_str: str) -> bool:
    """
    Verify tinycio version. Check if the major and minor version of `ver_str` matches current.

    :param ver_str: Version string to compare
    :return: True if major.minor match, else False
    """
    ver = TINYCIO_VERSION.split('.')
    chk = ver_str.split('.')
    return ver[0] == chk[0] and ver[1] == chk[1]

def remap(x: Union[float, torch.Tensor], from_start: float, from_end: float, to_start: float, to_end: float) -> Union[float, torch.Tensor]:
    """
    Linearly remap scalar or tensor.

    :param x: Input value or tensor
    :param from_start: Start of input range
    :param from_end: End of input range
    :param to_start: Start of target range
    :param to_end: End of target range
    :return: Remapped and clamped value
    """
    res = (x - from_start) / (from_end - from_start) * (to_end - to_start) + to_start
    return torch.clamp(res, to_start, to_end) if torch.is_tensor(res) else np.clip(res, to_start, to_end)

def remap_to_01(x: Union[float, torch.Tensor], start: float, end: float) -> Union[float, torch.Tensor]:
    """
    Remap value to [0, 1] range.

    :param x: Input value or tensor
    :param start: Start of original range
    :param end: End of original range
    :return: Normalized value clamped to [0, 1]
    """
    res = (x - start) / (end - start)
    return torch.clamp(res, 0., 1.) if torch.is_tensor(res) else np.clip(res, 0., 1.)

def remap_from_01(x: Union[float, torch.Tensor], start: float, end: float) -> Union[float, torch.Tensor]:
    """
    Remap [0, 1] value back to specified range.

    :param x: Normalized value or tensor
    :param start: Target range start
    :param end: Target range end
    :return: Rescaled value clamped to [start, end]
    """
    res = x * (end - start) + start
    return torch.clamp(res, start, end) if torch.is_tensor(res) else np.clip(res, start, end)

def smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """
    Smooth Hermite interpolation between 0 and 1. For x in [edge0, edge1].

    :param edge0: Lower bound of transition
    :param edge1: Upper bound of transition
    :param x: Input tensor
    :return: Smoothly interpolated tensor
    """
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def softsign(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    Smooth nonlinearity. x / (1 + \|x\|), useful for range compression.

    :param x: Input scalar or tensor
    :return: Softsign result
    """
    return x / (1 + x.abs()) if torch.is_tensor(x) else x / (1 + np.abs(x))

def fract(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    Get the fractional part of input (x - floor(x)).

    :param x: Input scalar or tensor
    :return: Fractional part
    """
    return x - torch.floor(x) if torch.is_tensor(x) else x - np.floor(x)

def serialize_tensor(val: torch.Tensor) -> Union[float, List[float]]:
    """
    Convert a tensor into a float or list of floats.

    :param val: Tensor to serialize
    :return: Scalar if 1-element tensor, else flattened list
    """
    if val.numel() == 1:
        return val.item()
    return val.flatten().tolist()

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