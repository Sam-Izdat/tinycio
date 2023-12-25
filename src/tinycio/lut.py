from __future__ import annotations
import typing
from typing import Union
import os 
from enum import IntEnum
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .colorspace import ColorSpace
from .fsio.lutfile import load_lut, save_lut, _infer_lut_file_format, _generate_linear_cube_lut
from .fsio.format import LUTFormat
from .util.colorutil import srgb_luminance
from .util.miscutil import trilinear_interpolation
from .loss import feature_moments_calculation

class LookupTable:
    """
    Color lookup table. Example:

    .. highlight:: python
    .. code-block:: python

        lut = LookupTable.get_negative()
        im_negative = lut.apply(im)

    :param size: Size of the LUT.
    :param lattice: Lattice as tensor (defaults to linear).
    :param lut_format: Format of the LUT.
    """
    size = 32
    lattice = None
    lut_format= LUTFormat.UNKNOWN

    __min_size, __max_size = 4, 512

    def __init__(self, size:int, lattice:torch.Tensor=None, lut_format:LUTFormat=LUTFormat.CUBE_3D):
        assert self.__min_size <= size <= self.__max_size, f"LUT size must be between {self.__min_size} and {self.__max_size}"
        self.size == size
        self.lattice = lattice if lattice is not None else _generate_linear_cube_lut(size)
        self.lut_format = lut_format

    @classmethod
    def load(cls, fp:str, lut_format:LUTFormat=LUTFormat.UNKNOWN) -> LookupTable:
        """
        Load LUT from file.

        :param fp: File path.
        :param lut_format: Format of the LUT.
        """
        fp = os.path.realpath(fp)
        fn, fnext = os.path.splitext(fp)

        variant = lut_format if lut_format > LUTFormat.UNKNOWN else _infer_lut_file_format(fnext)

        assert variant > LUTFormat.UNKNOWN, "Unrecognized LUT format"

        lattice = load_lut(fp, variant)

        return cls(lattice.size(0), lattice, variant)

    def save(self, fp:str, lut_format:LUTFormat=LUTFormat.UNKNOWN):
        """
        Save LUT to file.

        .. warning:: 
        
            This will overwrite existing files.

        :param fp: File path.
        :param lut_format: Format of the LUT.
        """
        fp = os.path.realpath(fp)
        fn, fnext = os.path.splitext(fp)

        variant = lut_format if lut_format > LUTFormat.UNKNOWN else _infer_lut_file_format(fnext) or self.variant

        assert variant > LUTFormat.UNKNOWN, "Unrecognized LUT format"

        lattice = save_lut(self.lattice, fp, variant)

        return True

    def apply(self, im:Union[torch.Tensor, ColorImage]) -> torch.Tensor:
        """
        Apply LUT to image tensor.

        :param im: Input image tensor
        :type im: torch.Tensor | ColorImage
        :return: Image tensor with LUT applied
        """
        assert self.lut_format > LUTFormat.UNKNOWN and self.lattice != None, "No LUT has been loaded"
        assert im.size(0) == 3, "Image should have three color channels (RGB)"
        assert self.lattice.size(-1) == 3, "Cube LUT should have three color channels"

        indices = (im * (self.lattice.size(0) - 1)).clamp(0, self.lattice.size(0) - 1)

        im_out = trilinear_interpolation(self.lattice, indices)
        return im_out

    @classmethod
    def get_linear(cls, size:int=32, lut_format:LUTFormat=LUTFormat.CUBE_3D) -> LookupTable:
        """
        Returns linear LUT. Has no effect: when applied, output matches input ([0, 1] range).

        :param size: Size of the LUT.
        :param lut_format: Format of the LUT.
        """
        if lut_format == LUTFormat.CUBE_3D:
            assert cls.__min_size <= size <= cls.__max_size, f"LUT size must be between {cls.__min_size} and {cls.__max_size}"
            variant = LUTFormat.CUBE_3D
            lattice = _generate_linear_cube_lut(size)
        else:
            raise Exception(f"Backpropagation not implemented for: {lut_format.name}")

        return cls(size, lattice, variant)

    @classmethod
    def get_negative(cls, size:int=32, lut_format:LUTFormat=LUTFormat.CUBE_3D) -> LookupTable:
        """
        Returns negative LUT. Output is inverted ([0, 1] range).

        :param size: Size of the LUT.
        :param lut_format: Format of the LUT.
        """
        lut = cls.get_linear(size, lut_format)
        lut.lattice = 1. - lut.lattice
        return lut

    @classmethod
    def get_random(cls, size:int=32, lut_format:LUTFormat=LUTFormat.CUBE_3D) -> LookupTable:
        """
        Returns random LUT. Everything mapped to random values ([0, 1] range).

        :param size: Size of the LUT.
        :param lut_format: Format of the LUT.
        """
        lut = cls.get_linear(size, lut_format)
        lut.lattice = torch.randn_like(lut.lattice)
        return lut

    @classmethod
    def get_empty(cls, size:int=32, lut_format:LUTFormat=LUTFormat.CUBE_3D) -> LookupTable:
        """
        Returns empty LUT. All values mapped to 0.

        :param size: Size of the LUT.
        :param lut_format: Format of the LUT.
        """
        lut = cls.get_linear(size, lut_format)
        lut.lattice = lut.lattice * 0.
        return lut

    def fit_to_image(self, 
        im_source:Union[torch.Tensor, ColorImage], 
        im_target:Union[torch.Tensor, ColorImage], 
        steps:int=500, 
        learning_rate:float=0.003,
        strength:float=1.,
        fit_height:int=512,
        fit_width:int=512,
        device:str='cuda',
        context:callable=None
        ) -> bool:
        """
        Perform gradient descent on the lattice, so that the appearance of the source image matches the target.

        :param im_source: Source image tensor. Values must be in range [0, 1].
        :type im_source: torch.Tensor | ColorImage
        :param im_target: Target image tensor.
        :type im_target: torch.Tensor | ColorImage
        :param steps: Number of optimization steps.
        :param learning_rate: Learning rate for gradient descent.
        :param strength: Strength of the effect in range [0, 1].
        :param fit_height: Image tensors will be interpolated to this height for evaluation.
        :param fit_width: Image tensors will be interpolated to this width for evaluation.
        :param device: Device for gradient descent (if None will use input tensor device).
        :return: True when completed
        """
        assert 0. <= strength <= 1., "strength must be in range [0, 1]"
        im_source = im_source.clone()
        device = torch.device(device.strip().lower()) if device is not None else im_source.device
        im_source = F.interpolate(
            im_source.unsqueeze(0), 
            size=[fit_height, fit_width], 
            mode='bicubic', 
            align_corners=False).squeeze(0).clamp(0.,1.).to(device)
        
        im_target = F.interpolate(
            im_target.unsqueeze(0), 
            size=[fit_height, fit_width], 
            mode='bicubic', 
            align_corners=False).squeeze(0).clamp(0.,1.).to(device)

        __ctx = context if context is not None and callable(context) else nullcontext
        with __ctx() as ctx:
            cb_callable = hasattr(ctx, 'update_fit_status') and callable(ctx.update_fit_status)
            cb = ctx.update_fit_status if cb_callable else lambda a, b, c, d: None
            if self.lut_format == LUTFormat.CUBE_3D:
                lut = torch.nn.Parameter(self.lattice)
                lut.requires_grad_()
                optimizer = optim.Adam([lut], lr=learning_rate)
                indices = (im_source * (lut.size(0) - 1)).clamp(0, lut.size(0) - 1).to(device)

                area = fit_height * fit_height
                fm_mean_scale = area
                fm_p2_scale = area / 32.
                fm_p3_scale = area / 64.
                selfsim_scale = area 
                sat_scale = area # lut optimization goes a bit wild with this
                for step in range(steps):
                    t_source = trilinear_interpolation(lut.to(device), indices).to(device)
                    loss = 0.

                    # Main feature loss
                    feat_source_mean, feat_source_p2, feat_source_p3 = feature_moments_calculation(t_source.view(1,3,-1))
                    feat_target_mean, feat_target_p2, feat_target_p3 = feature_moments_calculation(im_target.view(1,3,-1))
                    loss += F.mse_loss(feat_source_mean, feat_target_mean) * fm_mean_scale * strength 
                    loss += F.mse_loss(feat_source_p2, feat_target_p2) * fm_p2_scale * strength 
                    loss += F.mse_loss(feat_source_p3, feat_target_p3) * fm_p3_scale * strength 

                    # Additional saturation-focused loss
                    sat_s = srgb_luminance(t_source).repeat(3,1,1) - t_source
                    sat_t = srgb_luminance(im_target).repeat(3,1,1) - im_target
                    loss += F.mse_loss(sat_s.mean(), sat_t.mean()) * sat_scale * strength 
                    loss += F.mse_loss(sat_s.std(), sat_t.std()) * sat_scale * strength 

                    # Self-similarity loss
                    if strength < 1.: loss += F.mse_loss(t_source, im_source) * selfsim_scale * (1. - strength)

                    # Perform backpropagation and optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 10 == 0: cb(step // 10 + 1, 10, steps, loss.item())

                self.lattice = lut.detach().clamp(0.,1.)
            else:
                raise Exception(f"Backpropagation not implemented for: {self.lut_format.name}")

        return True