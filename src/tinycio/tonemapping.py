import torch
import numpy as np
import typing
from enum import IntEnum

from .np_agx.agx import applyAgX, applyAgXPunchy
from .colorspace import ColorSpace, TransferFunction

class ToneMapping:
    """
    Map high-dynamic-range values to low-dynamic-range. LDR is typically sRGB in [0, 1] range. Example:

    .. highlight:: python
    .. code-block:: python
        
        tm = ToneMapping.Variant.HABLE
        tonemapped_image = ToneMapping.apply(input_im, tone_mapper=tm)

    """
    class Variant(IntEnum):
        """
        Tone mapper enum. Available options are:

        .. highlight:: text
        .. code-block:: text
        
            - NONE
            - CLAMP
            - AGX
            - AGX_PUNCHY
            - HABLE
            - REINHARD
            - ACESCG
        """
        NONE                = 1<<0
        CLAMP               = 1<<1
        AGX                 = 1<<2
        AGX_PUNCHY          = 1<<3
        HABLE               = 1<<4
        REINHARD            = 1<<5
        ACESCG              = 1<<6

        IP_SRGB_LIN         = CLAMP | AGX | AGX_PUNCHY | HABLE | REINHARD 
        IP_ACESCG           = ACESCG

        OP_SRGB_LIN         = CLAMP | AGX | AGX_PUNCHY | HABLE | REINHARD 
        OP_ACESCG           = ACESCG

        DISABLED            = 0

    @classmethod
    def apply(cls, im:torch.Tensor, tone_mapper:Variant):
        """
        Apply tone mapping to HDR image tensor. Input data is expected to be in the correct color 
        space for the chosen tone mapper.

        .. note::

            :code:`ACESCG` tone mapping is performed on AP1 primaries and expects input in the 
            :code:`ACESCG` color space. All other tone mappers expect :code:`SRGB_LIN`.
            The :code:`tone_map()` method of :class:`ColorImage` handles this conversion automatically.

        :param torch.Tensor im: [C=3, H, W] sized image tensor 
        :param ToneMapping.Variant tone_mapper: tonemapper to be used
        :return: image tensor
        :rtype: torch.Tensor
        """
        assert im.dim() == 3 and im.size(0) == 3, f"expected [C=3, H, W] image tensor, got {im.size()}"
        op, tm = tone_mapper, cls.Variant
        err_not_supported, err_disabled = f"ToneMapping {op.name} is not supported", f"ToneMapping {op.name} is disabled"
        if op & tm.DISABLED: raise Exception(err_disabled)

        if   op == tm.NONE:         return im
        elif op == tm.CLAMP:        return im.clamp(0., 1.)
        elif op == tm.AGX:          return cls._agx(im)
        elif op == tm.AGX_PUNCHY:   return cls._agx_punchy(im)
        elif op == tm.HABLE:        return cls._hable(im)
        elif op == tm.REINHARD:     return cls._reinhard_extended_luminance(im)
        elif op == tm.ACESCG:       return cls._aces_fitted(im)
        else: raise Exception(err_not_supported)

        return out

    @classmethod
    def _agx(cls, im:torch.Tensor):
        device = im.device
        out = applyAgX(im.permute(1, 2, 0).cpu().numpy())
        out = torch.from_numpy(out).permute(2, 0, 1).to(device)
        return TransferFunction.srgb_eotf(out.clamp(0., 1.))

    @classmethod
    def _agx_punchy(cls, im:torch.Tensor):
        device = im.device
        out = applyAgXPunchy(im.permute(1, 2, 0).cpu().numpy())
        out = torch.from_numpy(out).permute(2, 0, 1).to(device)
        return TransferFunction.srgb_eotf(out.clamp(0., 1.))

    @classmethod
    def __luminance(cls, im:torch.Tensor):
        """
        Compute luminance of the input image.

        :param torch.Tensor im: Input image tensor.
        :return: Luminance tensor.
        :rtype: torch.Tensor
        """
        lum = torch.sum(im.permute(1,2,0) * torch.tensor([[0.2126, 0.7152, 0.0722]]), dim=2, keepdim=True)
        return lum.permute(2,0,1).repeat(3,1,1)

    @classmethod
    def __change_luminance(cls, c_in:torch.Tensor, l_out:torch.Tensor):
        """
        Change luminance of the input image.

        :param torch.Tensor c_in: Input image tensor.
        :param torch.Tensor l_out: Target luminance.
        :return: Image tensor with adjusted luminance.
        :rtype: torch.Tensor
        """
        l_in = cls.__luminance(c_in)
        return c_in * (l_out / l_in)

    @classmethod
    def __hable_partial(cls, im:torch.Tensor):
        """
        Partial Hable operator for tone mapping.

        :param torch.Tensor im: Input image tensor.
        :return: Partial Hable-transformed image tensor.
        :rtype: torch.Tensor
        """
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        return ((im*(A*im+C*B)+D*E)/(im*(A*im+B)+D*F))-E/F

    @classmethod
    def _hable(cls, im:torch.Tensor):
        """
        Apply Hable Filmic (Uncharted 2) tone mapping.

        :param torch.Tensor im: Input linear sRGB image tensor.
        :return: Tone-mapped linear sRGB image tensor.
        :rtype: torch.Tensor
        """
        exposure_bias = 2.
        curr = cls.__hable_partial(im * exposure_bias)
        w = 11.2
        white_scale = 1. / cls.__hable_partial(w)
        return torch.clamp(curr * white_scale, 0., 1.)

    @classmethod
    def _reinhard_extended_luminance(cls, im:torch.Tensor):
        """
        Apply Extended Reinhard tone mapping (Luminance).

        :param torch.Tensor im: Input linear sRGB image tensor.
        :return: Tone-mapped linear sRGB image tensor.
        :rtype: torch.Tensor
        """
        l_old = cls.__luminance(im)
        max_white_l = l_old.max().item()
        numerator = l_old * (1. + (l_old / (max_white_l * max_white_l)))
        l_new = numerator / (1. + l_old)
        return torch.clamp(cls.__change_luminance(im, l_new), 0., 1.)

    @classmethod
    def _aces_fitted(cls, im:torch.Tensor):
        """
        Apply ACES (Stephen Hill's fitted version) tone mapping.
        
        .. note::
        
            expects [C, H, W] input in ACESCcg color space, return ACEScg image tensor

        :param torch.Tensor im: Input image tensor.
        :return: Tone-mapped image tensor.
        :rtype: torch.Tensor
        """
        C, H, W = im.size()

        # RRT and ODT
        im = ColorSpace._ap1_rrt_sat(im)
        a = im * (im + 0.0245786) - 0.000090537
        b = im * (0.983729 * im + 0.4329510) + 0.238081
        im  = ColorSpace._ap1_odt_sat(a / b)

        return torch.clamp(im, 0., 1.) 