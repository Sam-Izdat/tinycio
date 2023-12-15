from __future__ import annotations
import typing
from typing import Union
import torch

class Codec:
    """
    Encoder/decoder for high-dynamic-range data. Example:

    .. highlight:: python
    .. code-block:: python

       encoded_image = Codec.logluv_encode(hdr_rgb_image)
    """
    mat_logluv_enc = [
        [0.2209, 0.3390, 0.4184],
        [0.1138, 0.6780, 0.7319],
        [0.0102, 0.1130, 0.2969]]

    mat_logluv_dec = [
        [6.0013, -2.700, -1.7995],
        [-1.332, 3.1029, -5.7720],
        [.3007, -1.088, 5.6268]]

    @classmethod
    def logluv_encode(cls, im:Union[torch.Tensor, ColorImage]) -> torch.Tensor:
        """
        Encode HDR floating point RGB data to LOGLUV for 32bpp (RGBA) image file. 

        .. note::
            3 channels in, 4 channels out.

        :param im: HDR [C=3, H, W] sized image tensor
        :type im: torch.Tensor | ColorImage
        :return: LOGLUV encoded image tensor (4 channels)
        """
        assert im.dim() == 3, f"expected [C=3, H, W] image tensor, got {im.size()}"
        C, H, W = im.size()
        assert C == 3, "HDR image needs 3 channels"
        src = im.clone().permute(1,2,0).reshape(-1, 1, 3)
        mat = torch.tensor(cls.mat_logluv_enc).unsqueeze(0).repeat(src.size(0), 1, 1).to(im.device)
        Xp_Y_XYZp = torch.bmm(src, mat)
        Xp_Y_XYZp = torch.maximum(Xp_Y_XYZp, torch.tensor([1e-6, 1e-6, 1e-6]))
        dst = torch.zeros(H*W, 1, 4).to(im.device)
        dst[..., 0:2] = Xp_Y_XYZp[..., 0:2] / Xp_Y_XYZp[..., 2:3]
        Le = 2 * torch.log2(Xp_Y_XYZp[..., 1]) + 127
        dst[..., 3] = torch.frac(Le);
        dst[..., 2] = (Le - (torch.floor(dst[..., 3] * 255)) / 255) / 255
        return dst.permute(2,0,1).view(4, H, W)

    @classmethod
    def logluv_decode(cls, im:torch.Tensor) -> torch.Tensor:
        """
        Decode LOGLUV to HDR floating point RGB data. 

        .. note::
            4 channels in, 3 channels out.

        :param im: LOGLUV encoded [C=4, H, W] sized image tensor (4 channels)
        :type im: torch.Tensor
        :return: Decoded HDR image tensor (3 channels)
        """
        assert im.dim() == 3, f"expected [C=4, H, W] image tensor, got {im.size()}"
        C, H, W = im.size()
        assert C == 4, "Logluv image needs 4 channels"
        src = im.clone().permute(1,2,0).reshape(-1, 1, 4)
        mat = torch.tensor(cls.mat_logluv_dec).unsqueeze(0).repeat(src.size(0), 1, 1).to(im.device)
        Le = src[..., 2] * 255 + src[..., 3]
        Xp_Y_XYZp = torch.zeros(H*W, 1, 3).to(im.device)
        Xp_Y_XYZp[..., 1] = torch.exp2((Le - 127) / 2)
        Xp_Y_XYZp[..., 2] = Xp_Y_XYZp[..., 1] / src[..., 1]
        Xp_Y_XYZp[..., 0] = src[..., 0] * Xp_Y_XYZp[..., 2]
        dst = torch.bmm(Xp_Y_XYZp, mat)
        return torch.maximum(dst, torch.zeros_like(dst).to(im.device)).permute(2,0,1).view(3, H, W)