from __future__ import annotations
import typing
from typing import Union
from contextlib import nullcontext
import os
import toml

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .numerics import Float2, Float3, saturate
from .util.colorutil import apply_hue_oklab, srgb_luminance, col_rgb_to_hsv, col_hsv_to_rgb, col_okhsv_to_srgb
from .util.miscutil import version_check_minor
from .colorspace import ColorSpace
from .lut import LookupTable
from .fsio.format import LUTFormat
from .loss import feature_moments_calculation
from .globals import TINYCIO_VERSION

class ColorCorrection:
    """    
    Apply color correction to an image. Example:

    .. highlight:: python
    .. code-block:: python

        cc = ColorCorrection()
        cc.set_contrast(1.3)
        im_corrected = cc.apply(im_cc)

    .. note::
        Any *hue* and *saturation* parameters use perceptually linear values of the OKHSV color space.
    """

    __contrast_midpoint         = 0.5
    __eps                       = 1e-5

    __range_exposure_bias       = (-5., +5.)
    __range_color_filter_h      = (+0., +1.)
    __range_color_filter_s      = (+0., +1.)
    __range_hue_delta           = (-1., +1.)
    __range_saturation          = (+0., +4.)
    __range_contrast            = (+0., +4.)
    __range_shadow_color_h      = (+0., +1.)
    __range_shadow_color_s      = (+0., +1.)
    __range_midtone_color_h     = (+0., +1.)
    __range_midtone_color_s     = (+0., +1.)
    __range_highlight_color_h   = (+0., +1.)
    __range_highlight_color_s   = (+0., +1.)
    __range_shadow_offs         = (-2., +2.)
    __range_midtone_offs        = (-2., +2.)
    __range_highlight_offs      = (-2., +2.)

    def __init__(self):

        self.exposure_bias      = 0.
        self.color_filter       = Float3(1.)
        self.hue_delta          = 0.
        self.saturation         = 1.

        self.contrast           = 1.


        self.shadow_color       = Float3(0.)
        self.midtone_color      = Float3(0.)
        self.highlight_color    = Float3(0.)

        self.shadow_offset      = 0.
        self.midtone_offset     = 0.
        self.highlight_offset   = 0.


    def apply(self, im:torch.Tensor) -> torch.Tensor:
        """
        Apply color correction to image tensor in ACEScc color space.

        :param im: Image tensor sized [C=3, H, W] in ACEScc color space.
        :return: Color corrected image tensor.
        """

        return self.__eval_all(im)

    def fit_to_image(self,
        im_source:Union[torch.Tensor, ColorImage], 
        im_target:Union[torch.Tensor, ColorImage], 
        steps:int=500, 
        learning_rate:float=0.003,
        strength:float=1.,
        allow_hue_shift:bool=False,
        fit_height:int=512,
        fit_width:int=512,
        device:str='cuda',
        context:callable=None
        )  -> ColorCorrection:
        """
        Perform gradient descent on the color correction settings, so that the appearance  
        of the source image matches the target. 

        .. note::
            Images need to be in *ACEScc* color space.

        :param im_source: Source image tensor in *ACEScc* color space. Values must be in range [0, 1].
        :type im_source: torch.Tensor | ColorImage
        :param im_target: Target image tensor in *ACEScc* color space.
        :type im_target: torch.Tensor | ColorImage
        :param steps: Number of optimization steps.
        :param learning_rate: Learning rate for gradient descent.
        :param strength: Strength of the effect in range [0, 1].
        :param allow_hue_shift: Allow the optimizer to shift the hue.
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
        r_exp   = self.__range_exposure_bias
        r_hue   = self.__range_hue_delta
        r_con   = self.__range_contrast
        r_sat   = self.__range_saturation
        r_sho   = self.__range_shadow_offs
        r_mio   = self.__range_midtone_offs
        r_hio   = self.__range_highlight_offs

        with __ctx() as ctx: #, torch.autograd.set_detect_anomaly(True):
            cb_callable = hasattr(ctx, 'update_fit_status') and callable(ctx.update_fit_status)
            cb = ctx.update_fit_status if cb_callable else lambda a, b, c, d: None

            cc_ten = torch.zeros(19, dtype=torch.float32)
            cc_ten[0] = float(self.exposure_bias)
            cc_ten[1] = float(self.color_filter.r)
            cc_ten[2] = float(self.color_filter.g)
            cc_ten[3] = float(self.color_filter.b)
            cc_ten[4] = float(self.hue_delta)
            cc_ten[5] = float(self.saturation)
            cc_ten[6] = float(self.contrast)
            cc_ten[7] = float(self.shadow_color.r)
            cc_ten[8] = float(self.shadow_color.g)
            cc_ten[9] = float(self.shadow_color.b)
            cc_ten[10] = float(self.midtone_color.r)
            cc_ten[11] = float(self.midtone_color.g)
            cc_ten[12] = float(self.midtone_color.b)
            cc_ten[13] = float(self.highlight_color.r)
            cc_ten[14] = float(self.highlight_color.g)
            cc_ten[15] = float(self.highlight_color.b)
            cc_ten[16] = float(self.shadow_offset)
            cc_ten[17] = float(self.midtone_offset)
            cc_ten[18] = float(self.highlight_offset)
            ccs = torch.nn.Parameter(cc_ten)
            ccs.requires_grad_()
            optimizer = optim.Adam([ccs], lr=learning_rate)

            area = fit_height * fit_height
            fm_mean_scale = area
            fm_p2_scale = area / 32.
            fm_p3_scale = area / 64.
            selfsim_scale = area
            sat_scale = area / 2.

            for step in range(steps):
                t_source = im_source.clone().to(device)
                t_source = self.__eval_exposure(t_source, 
                    exposure_bias=ccs[0].view(1).to(device).clamp(r_exp[0], r_exp[1]), 
                    color_filter=(
                        ccs[1].view(1).to(device).clamp(0., 1.), 
                        ccs[2].view(1).to(device).clamp(0., 1.), 
                        ccs[3].view(1).to(device).clamp(0., 1.)))
                if allow_hue_shift:
                    t_source = self.__eval_hue(t_source, 
                        hue_delta=ccs[4].view(1).to(device).clamp(r_hue[0], r_hue[1]))
                t_source = self.__eval_saturation(t_source, 
                    saturation=ccs[5].view(1).to(device).clamp(r_sat[0], r_sat[1]))
                t_source = self.__eval_contrast(t_source, 
                    contrast=ccs[6].view(1).to(device).clamp(r_con[0], r_con[1]))
                t_source = self.__eval_lift_gamma_gain(t_source,
                    shc=(ccs[7].view(1).to(device).clamp(0., 1.), 
                        ccs[8].view(1).to(device).clamp(0., 1.), 
                        ccs[9].view(1).to(device).clamp(0., 1.)),
                    mic=(ccs[10].view(1).to(device).clamp(0., 1.), 
                        ccs[11].view(1).to(device).clamp(0., 1.), 
                        ccs[12].view(1).to(device).clamp(0., 1.)), 
                    hic=(ccs[13].view(1).to(device).clamp(0., 1.), 
                        ccs[14].view(1).to(device).clamp(0., 1.), 
                        ccs[15].view(1).to(device).clamp(0., 1.)),
                    sho=ccs[16].view(1).to(device).clamp(r_sho[0], r_sho[1]), 
                    mio=ccs[17].view(1).to(device).clamp(r_mio[0], r_mio[1]), 
                    hio=ccs[18].view(1).to(device).clamp(r_hio[0], r_hio[1])
                    )

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

            res = ccs.detach()

            self.exposure_bias = res[0].clamp(r_exp[0], r_exp[1]).item()
            self.color_filter = Float3(
                res[1].clamp(0., 1.).item(), 
                res[2].clamp(0., 1.).item(), 
                res[3].clamp(0., 1.).item())
            self.hue_delta = res[4].clamp(r_hue[0], r_hue[1]).item()
            self.saturation = res[5].clamp(r_sat[0], r_sat[1]).item()
            self.contrast = res[6].clamp(r_con[0], r_con[1]).item()
            self.shadow_color = Float3(
                res[7].clamp(0., 1.).item(),
                res[8].clamp(0., 1.).item(),
                res[9].clamp(0., 1.).item())
            self.midtone_color = Float3(
                res[10].clamp(0., 1.).item(),
                res[11].clamp(0., 1.).item(),
                res[12].clamp(0., 1.).item())
            self.highlight_color    = Float3(
                res[13].clamp(0., 1.).item(),
                res[14].clamp(0., 1.).item(),
                res[15].clamp(0., 1.).item())
            self.shadow_offset = res[16].clamp(r_sho[0], r_sho[1]).item()
            self.midtone_offset = res[17].clamp(r_mio[0], r_mio[1]).item()
            self.highlight_offset = res[18].clamp(r_hio[0], r_hio[1]).item()

            return self

    def save(self, fp:str) -> bool:
        """
        Save color correction settings to a *toml* file.
        
        :param fp: Output file path of *toml* file to be saved.
        :return: True if successful.

        .. warning:: 
            This will overwrite existing files.
        """
        fp = os.path.realpath(fp)
        data = {
            "domain": "tinycio",
            "tinycio_version": TINYCIO_VERSION,
            "format": __class__.__name__,
            "cc_settings": {
                "exposure_bias":    float(self.exposure_bias),
                "color_filter":     [
                    float(self.color_filter[0]),
                    float(self.color_filter[1]),
                    float(self.color_filter[2])],
                "hue_delta":        float(self.hue_delta),
                "saturation":       float(self.saturation),
                "contrast":         float(self.contrast),
                "shadow_color":     [
                    float(self.shadow_color[0]),
                    float(self.shadow_color[1]),
                    float(self.shadow_color[2])],
                "midtone_color":    [
                    float(self.midtone_color[0]),
                    float(self.midtone_color[1]),
                    float(self.midtone_color[2])],
                "highlight_color":  [
                    float(self.highlight_color[0]),
                    float(self.highlight_color[1]),
                    float(self.highlight_color[2])],
                "shadow_offset":    float(self.shadow_offset),
                "midtone_offset":   float(self.midtone_offset),
                "highlight_offset": float(self.highlight_offset)
            }
        }
        with open(fp, "w") as tf:
            toml.dump(data, tf)

        return True

    @classmethod
    def load(cls, fp:str, force:bool=False) -> ColorCorrection:
        """
        Load color correction settings from *toml* file.
        
        :param fp: File path of *toml* file to be loaded.
        :param force: If set to True, ignores version mismatch forces loading the file.
        :return: :class:`.ColorCorrection` object with settings loaded.
        """
        fp = os.path.realpath(fp)
        with open(fp) as tf:
            data = toml.load(tf)

            domain = str(data['domain'])
            dfmt = str(data['format'])
            version_check = True
            if not force: version_check = version_check_minor(str(data['tinycio_version']))
            assert domain == "tinycio", "unrecognized toml file"
            assert version_check, f"minor version mismatch; expected {TINYCIO_VERSION}, got {data['tinycio_version']}"
            assert dfmt == __class__.__name__, "unrecognized data format"

            cc = cls()
            settings = data['cc_settings']

            # no setters on precomputed color 
            cc.set_exposure_bias(float(settings['exposure_bias']))
            cc.color_filter     = Float3(
                float(settings['color_filter'][0]),
                float(settings['color_filter'][1]),
                float(settings['color_filter'][2])).clip(0., 1.)
            cc.set_hue_delta(float(settings['hue_delta']))
            cc.set_saturation(float(settings['saturation']))
            cc.set_contrast(float(settings['contrast']))
            cc.shadow_color     = Float3(
                float(settings['shadow_color'][0]),
                float(settings['shadow_color'][1]),
                float(settings['shadow_color'][2])).clip(0., 1.)
            cc.midtone_color    = Float3(
                float(settings['midtone_color'][0]),
                float(settings['midtone_color'][1]),
                float(settings['midtone_color'][2])).clip(0., 1.)
            cc.highlight_color  = Float3(
                float(settings['highlight_color'][0]),
                float(settings['highlight_color'][1]),
                float(settings['highlight_color'][2],)).clip(0., 1.)
            cc.set_shadow_offset(float(settings['shadow_offset']))
            cc.set_midtone_offset(float(settings['midtone_offset']))
            cc.set_highlight_offset(float(settings['highlight_offset']))

            return cc

    def info(self, printout:bool=True) -> Union[bool, str]:
        """
        Print or return a string describing the current color correction settings.

        :param printout: Print string if True, return string if False
        """
        infostr = "CC DESCRIPTION:"
        infostr += "\n==============="
        infostr += "\nCLASS            " + str(self.__class__.__name__)
        infostr += "\nEXPOSURE BIAS    " + str(self.exposure_bias)
        infostr += "\nCOLOR FILTER     " + str(self.color_filter)
        infostr += "\nHUE DELTA        " + str(self.hue_delta)
        infostr += "\nSATURATION       " + str(self.saturation)
        infostr += "\nCONTRAST         " + str(self.contrast)
        infostr += "\nSHADOW COLOR     " + str(self.shadow_color)
        infostr += "\nMIDTONE COLOR    " + str(self.midtone_color)
        infostr += "\nHIGHLIGHT COLOR  " + str(self.highlight_color)
        infostr += "\nSHADOW OFFSET    " + str(self.shadow_offset)
        infostr += "\nMIDTONE OFFSET   " + str(self.midtone_offset)
        infostr += "\nHIGHLIGHT OFFSET " + str(self.highlight_offset)
        if printout: 
            print(infostr)
            return
        return infostr

    def bake_lut(self, 
        lut_size:int=64, 
        lut_color_space:Union[str, ColorSpace.Variant]=ColorSpace.Variant.ACESCC) -> LookupTable:
        """
        Bake color correction to a CUBE LUT. 

        .. note::
            Regardless of working color space, the LUT will be limited to a [0, 1] range of values.

        :param lut_size: Size of the LUT. Range [4, 512].
        :param lut_color_space: The color space that the LUT should be baked for, as string or IntEnum.
        :return: The baked LUT.
        """
        assert 4 <= lut_size <= 512, "lut_size must be in range [4, 512]"

        lut = LookupTable.get_linear(lut_size)
        lattice = lut.lattice.permute(3, 0, 1, 2)
        cs_lut = ColorSpace.Variant[lut_color_space] if type(lut_color_space) == str else lut_color_space
        cs_cc = ColorSpace.Variant.ACESCC
        for d in range(lut_size):
            slice_cc = ColorSpace.convert(lattice[..., d], cs_lut, cs_cc)
            lattice[..., d] = ColorSpace.convert(self.apply(slice_cc), cs_cc, cs_lut)

        lattice = lattice.permute(1, 2, 3, 0).clamp(0., 1.)
        lut_out = LookupTable(size=lut_size, lattice=lattice, lut_format=LUTFormat.CUBE_3D)
        return lut_out


    def set_color_filter(self, hue:float=0., saturation:float=0.) -> bool:
        """
        Set a color filter. Idempotent assignment.

        :param hue: Filter hue - perceptually linear (the H from OKHSV). Range [0, 1].
        :param saturation: Filter saturation - perceptually linear (the S from OKHSV).
            Range [0, 1].
        :return: True if successful
        """
        cfhr = self.__range_color_filter_h
        cfsr = self.__range_color_filter_s
        assert cfhr[0] <= hue <= cfhr[1], f"hue must be in range [{cfhr[0]}, {cfhr[1]}]"
        assert cfsr[0] <= saturation <= cfsr[1], f"saturation must be in range [{cfsr[0]}, {cfsr[1]}]"
        color_filter_srgb = col_okhsv_to_srgb(Float3(hue, saturation, 1.))
        self.color_filter = color_filter_srgb
        return True

    def set_exposure_bias(self, exposure_bias:float=0.) -> bool:
        """
        Set the exposure bias. Idempotent assignment.

        :param exposure_bias: Exposure bias in f-stops. Range [-5, 5].
        :return: True if successful
        """
        ebr = self.__range_exposure_bias
        assert ebr[0] <= exposure_bias <= ebr[1], f"exposure_bias must be in range [{ebr[0]}, {ebr[1]}]"
        self.exposure_bias = exposure_bias
        return True

    def set_hue_delta(self, hue_delta:float=0.) -> bool:
        """
        Set the hue delta, shifting an image's hues. Idempotent assignment.

        :param hue_delta: Amount of hue shift - perceptually linear. Range [-1, 1].
        :return: True if successful
        """
        hdr = self.__range_exposure_bias
        assert hdr[0] <= hue_delta <= hdr[1], f"hue delta must be in range [{hdr[0]}, {hdr[1]}]"
        self.hue_delta = hue_delta
        return True

    def set_saturation(self, saturation:float=1.) -> bool:
        """
        Set the color saturation. Idempotent assignment.

        :param saturation: Amount of saturation. Range [0, 4].
        :return: True if successful
        """
        sr = self.__range_saturation
        assert sr[0] <= saturation <= sr[1], f"saturation must be in range [{sr[0]}, {sr[1]}]"
        self.saturation = saturation
        return True

    def set_contrast(self, contrast:float=1.) -> bool:
        """
        Set the contrast. Idempotent assignment.

        :param contrast: Contrast. Range [0, 4].
        :return: True if successful
        """
        cr = self.__range_contrast
        assert cr[0] <= contrast <= cr[1], f"contrast must be in range [{cr[0]}, {cr[1]}]"
        self.contrast = contrast
        return True

    def set_shadow_color(self, hue:float=0., saturation:float=0.) -> bool:
        """
        Set the shadow color. Uses OKHSV model - value fixed at 1. Idempotent assignment.
        
        :param hue: Shadow hue in range [0, 1] - perceptually linear (the H from OKHSV)
        :param saturation: Shadow saturation  in range [0, 1] - perceptually linear (the S from OKHSV)
        :return: True if successful
        """
        schr = self.__range_shadow_color_h
        scsr = self.__range_shadow_color_s
        assert schr[0] <= hue <= schr[1], f"shadow color hue must be in range [{schr[0]}, {schr[1]}]"
        assert scsr[0] <= saturation <= scsr[1], f"shadow color saturation must be in range [{scsr[0]}, {scsr[1]}]"
        shadow_color_srgb = col_okhsv_to_srgb(Float3(hue, saturation, 1.))
        self.shadow_color = shadow_color_srgb
        return True

    def set_midtone_color(self, hue:float=0., saturation:float=0.) -> bool:
        """
        Set the midtone color. Uses OKHSV model - value fixed at 1. Idempotent assignment.
        
        :param hue: Midtone hue in range [0, 1]  - perceptually linear (the H from OKHSV)
        :param saturation: Midtone saturation  in range [0, 1] - perceptually linear (the S from OKHSV)
        :return: True if successful
        """
        mihr = self.__range_midtone_color_h
        misr = self.__range_color_filter_s
        assert mihr[0] <= hue <= mihr[1], f"midtone color hue must be in range [{mihr[0]}, {mihr[1]}]"
        assert misr[0] <= saturation <= misr[1], f"midtone color saturation must be in range [{misr[0]}, {misr[1]}]"
        midtone_color_srgb = col_okhsv_to_srgb(Float3(hue, saturation, 1.))
        self.midtone_color = midtone_color_srgb
        return True

    def set_highlight_color(self, hue:float=0., saturation:float=0.) -> bool:
        """
        Set the highlight color. Uses OKHSV model - value fixed at 1. Idempotent assignment.
        
        :param hue: Highlight hue in range [0, 1]  - perceptually linear (the H from OKHSV)
        :param saturation: Highlight saturation in range [0, 1]  - perceptually linear (the S from OKHSV)
        :return: True if successful
        """
        hihr = self.__range_shadow_color_h
        hisr = self.__range_shadow_color_s
        assert hihr[0] <= hue <= hihr[1], f"highlight hue must be in range [{hihr[0]}, {hihr[1]}]"
        assert hisr[0] <= saturation <= hisr[1], f"highlight saturation must be in range [{hisr[0]}, {hisr[1]}]"
        highlight_color_srgb = col_okhsv_to_srgb(Float3(hue, saturation, 1.))
        self.highlight_color = highlight_color_srgb
        return True

    # TODO: I don't know what acceptable ranges here would be or how to make the descriptions useful.
    def set_shadow_offset(self, offset:float=0.) -> bool:
        """
        Set the shadow offset in range [-2, 2]. Idempotent assignment.
        
        :param offset: Shadow offset.
        :return: True if successful
        """
        shor = self.__range_shadow_offs
        assert shor[0] <= offset <= shor[1], f"shadow offset must be in range [{shor[0]}, {shor[1]}]"
        self.shadow_offset = offset
        return True

    def set_midtone_offset(self, offset:float=0.) -> bool:
        """
        Set the shadow offset in range [-2, 2]. Idempotent assignment.
        
        :param offset: Midtone offset.
        :return: True if successful
        """
        mior = self.__range_midtone_offs
        assert mior[0] <= offset <= mior[1], f"midtone offset must be in range [{mior[0]}, {mior[1]}]"
        self.midtone_offset = offset
        return True

    def set_highlight_offset(self, offset:float=0.) -> bool:
        """
        Set the shadow offset in range [-2, 2]. Idempotent assignment.
        
        :param offset: Highlight offset.
        :return: True if successful
        """
        hior = self.__range_highlight_offs
        assert hior[0] <= offset <= hior[1], f"highlight offset must be in range [{hior[0]}, {hior[1]}]"
        self.highlight_offset = offset
        return True

    # Redundant, but potentially needed as usertypes shan't be imported
    def __srgb_to_acescc(self, srgb:Float3) -> Float3:        
        cs_srgb_lin = ColorSpace.Variant.SRGB_LIN
        cs_acc = ColorSpace.Variant.ACESCC
        ten = torch.from_numpy(srgb)
        ten = ten.unsqueeze(-1).unsqueeze(-1)
        ten = ColorSpace.convert(ten, source=cs_srgb_lin, destination=cs_acc)
        return Float3(ten)

    def __acescg_to_acescc(self, cg:torch.Tensor) -> torch.Tensor:
        res = torch.where(cg < 0.00003051757, 
            (torch.log2(0.00001525878 + cg * 0.5) + 9.72) / 17.52, 
            (torch.log2(cg) + 9.72) / 17.52)
        return res

    def __acescc_to_acescg(self, cc:torch.Tensor) -> torch.Tensor:
        res = torch.where(cc < -0.3013698630, 
            (torch.exp2(cc * 17.52 - 9.72) - 0.00001525878) * 2,
            torch.exp2(cc * 17.52 - 9.72))
        return res

    def __eval_all(self, im:torch.Tensor, settings_tensor=None):
        im = im.clone()
        f3_one =  Float3(1.)

        if settings_tensor is not None:
            pass
        else:
            if self.exposure_bias != 0. or not np.array_equal(self.color_filter, f3_one):
                im = self.__eval_exposure(im)

            if self.hue_delta != 0.:    im = self.__eval_hue(im)
            if self.saturation != 1.:   im = self.__eval_saturation(im)
            if self.contrast != 1.:     im = self.__eval_contrast(im)

            if not np.array_equal(self.shadow_color, f3_one) \
                or not np.array_equal(self.midtone_color, f3_one) \
                or not np.array_equal(self.highlight_color, f3_one) \
                or self.shadow_offset != 0. or self.midtone_offset != 0. or self.highlight_offset != 0.:
                im = self.__eval_lift_gamma_gain(im)
        return im

    def __eval_exposure(self, 
        im:torch.Tensor, 
        exposure_bias:torch.Tensor=None, 
        color_filter:tuple=None) -> torch.Tensor:
        if exposure_bias is None: exposure_bias = self.exposure_bias
        color_filter = torch.cat(color_filter, dim=0).to(im.device) if color_filter is not None else self.color_filter

        im = self.__acescc_to_acescg(im)
        im = im * self.__ef3(color_filter, im) * (2. ** exposure_bias)
        return self.__acescg_to_acescc(im)

    def __eval_hue(self, im:torch.Tensor, hue_delta:torch.Tensor=None) -> torch.Tensor:
        if hue_delta is None: hue_delta = self.hue_delta
        cs_srgb_lin = ColorSpace.Variant.SRGB_LIN
        cs_oklab = ColorSpace.Variant.OKLAB
        im = ColorSpace.convert(im, source=cs_srgb_lin, destination=cs_oklab)
        im = apply_hue_oklab(im, hue_delta)
        im = ColorSpace.convert(im, source=cs_oklab, destination=cs_srgb_lin)
        return im
        
    def __eval_saturation(self, im:torch.Tensor, saturation:torch.Tensor=None) -> torch.Tensor:
        if saturation is None: saturation = self.saturation
        luma = srgb_luminance(im) # TODO: should this be ACEScc-specific?
        luma = luma.repeat(3,1,1)
        return luma + saturation * (im - luma)

    def __eval_contrast(self, im:torch.Tensor, contrast:torch.Tensor=None) -> torch.Tensor:
        if contrast is None: contrast = self.contrast 
        im[0] = self.__comp_log_contrast(im[0], contrast)
        im[1] = self.__comp_log_contrast(im[1], contrast)
        im[2] = self.__comp_log_contrast(im[2], contrast)
        return im

    def __eval_lift_gamma_gain(self, im:torch.Tensor, 
        shc:tuple=None, 
        mic:tuple=None, 
        hic:tuple=None, 
        sho:torch.Tensor=None, 
        mio:torch.Tensor=None, 
        hio:torch.Tensor=None) -> torch.Tensor:
        shadow_color        = torch.cat(shc, dim=0).to(im.device) if shc is not None else self.shadow_color
        midtone_color       = torch.cat(mic, dim=0).to(im.device) if mic is not None else self.midtone_color
        highlight_color     = torch.cat(hic, dim=0).to(im.device) if hic is not None else self.highlight_color

        shadow_offset       = sho if sho is not None else self.shadow_offset
        midtone_offset      = mio if mio is not None else self.midtone_offset
        highlight_offset    = hio if hio is not None else self.highlight_offset

        c_lift  = shadow_color - shadow_color.mean()
        c_gamma = midtone_color - midtone_color.mean()
        c_gain  = highlight_color - highlight_color.mean()
        eps = self.__eps
        mid_grey = (0.5 + c_gamma + midtone_offset) + eps

        adj_lift = 0. + c_lift + shadow_offset
        adj_gain = 1. + c_gain + highlight_offset
        adj_gamma = midtone_color * 0. # for autograd

        adj_gamma[0] = (self.__epsat((0.5 - adj_lift[0]) / (adj_gain[0] - adj_lift[0]))) / self.__epsat(mid_grey[0])
        adj_gamma[1] = (self.__epsat((0.5 - adj_lift[1]) / (adj_gain[1] - adj_lift[1]))) / self.__epsat(mid_grey[1])
        adj_gamma[2] = (self.__epsat((0.5 - adj_lift[2]) / (adj_gain[2] - adj_lift[2]))) / self.__epsat(mid_grey[2])

        im_c1, im_c2, im_c3 = im[0], im[1], im[2] # necessary to avoid in-place operation
        im_c1 = self.__comp_lift_gamm_gain(im_c1, adj_lift[0], 1./adj_gamma[0], adj_gain[0])
        im_c2 = self.__comp_lift_gamm_gain(im_c2, adj_lift[1], 1./adj_gamma[1], adj_gain[1])
        im_c3 = self.__comp_lift_gamm_gain(im_c3, adj_lift[2], 1./adj_gamma[2], adj_gain[2])

        return torch.stack([im_c1, im_c2, im_c3], dim=0)

    def __comp_log_contrast(self, im:torch.Tensor, contrast:Union[float, torch.Tensor]) -> torch.Tensor:
        adj_x = self.__contrast_midpoint + (im - self.__contrast_midpoint) * contrast
        return torch.maximum(im * 0., (adj_x) - self.__eps)

    def __comp_contrast_rev(self, im:torch.Tensor) -> torch.Tensor:
        log_x = im + self.__eps
        adj_x = (log_x - self.__contrast_midpoint)/self.contrast + self.__contrast_midpoint;
        return torch.maximum(im * 0., adj_x - self.__eps)
    
    def __comp_offset_power_slope(self, im:torch.Tensor, offset:float, power:float, slope:float) -> torch.Tensor:
        so = im * slope + offset
        # so = torch.where(so > 0., torch.pow(so, power), so)
        so[so > 0] = torch.pow(so[so > 0], power)
        return so

    def __comp_lift_gamm_gain(self, im, lift:float, inv_gamma:float, gain:float) -> torch.Tensor:
        power = inv_gamma
        offset = lift * gain
        slope = (1. - lift) * gain
        return self.__comp_offset_power_slope(im, offset, power, slope)

    def __epsat(self, val):
        res = torch.maximum(saturate(val), val * 0. + self.__eps) if torch.is_tensor(val) \
            else np.maximum(saturate(val), self.__eps)
        return res

    def __ef3(self, v:Union[Float3, torch.Tensor], ten:torch.Tensor) -> torch.Tensor:
        """Expand as torch tensor"""
        if not torch.is_tensor(v): v = torch.from_numpy(v)
        return v.unsqueeze(-1).unsqueeze(-1).expand_as(ten)