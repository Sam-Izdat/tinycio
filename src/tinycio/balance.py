from __future__ import annotations
import typing
from typing import Union
import torch
import numpy as np
from enum import IntEnum

from .colorspace import ColorSpace
from .numerics import Float2, matmul_tl as mm

class WhiteBalance:
    """    
    Adjust white balance of an image. Example:

    .. highlight:: python
    .. code-block:: python

       source_white = WhiteBalance.wp_from_image(input_image)
       target_white = WhiteBalance.wp_from_illuminant(WhiteBalance.Illuminant.NORTH_SKY)
       white_balanced_image = WhiteBalance.apply(input_image, source_white, target_white)
    """
    class Illuminant(IntEnum):
        """
        CIE 1931 2° standard illuminant. Available options are:

        .. highlight:: text
        .. code-block:: text
        
            - NONE
            - A (INCANDESCENT, TUNGSTEN) 
            - D50 (HORIZON)
            - D55 (MIDMORNING)
            - D65 (DAYLIGHT_NOON)
            - D75 (NORTH SKY)
            - D93 (BLUE_PHOSPHOR)
            - E (EQUAL_ENERGY)
            - F1 (FLUORESCENT_DAYLIGHT1)
            - F2 (FLUORESCENT_COOL_WHITE)
            - F3 (FLUORESCENT_WHITE)
            - F4 (FLUORESCENT_WARM_WHITE)
            - F5 (FLUORESCENT_DAYLIGHT2)
            - F6 (FLUORESCENT_LIGHT_WHITE)
            - F7 (D65_SIMULATOR, DAYLIGHT_SIMULATOR)
            - F8 (D50_SIMULATOR, SYLVANIA_F40)
            - F9 (FLOURESCENT_COOL_WHITE_DELUXE)
            - F10 (PHILIPS_TL85, ULTRALUME_50)
            - F11 (PHILIPS_TL84, ULTRALUME_40)
            - F12 (PHILIPS_TL83, ULTRALUME_30)
            - LED_B1 (PHOSPHOR_CONVERTED_BLUE1)
            - LED_B2 (PHOSPHOR_CONVERTED_BLUE2)
            - LED_B3 (PHOSPHOR_CONVERTED_BLUE3)
            - LED_B4 (PHOSPHOR_CONVERTED_BLUE4)
            - LED_B5 (PHOSPHOR_CONVERTED_BLUE5)
            - LED_BH1
            - LED_RGB1
            - LED_V1 (LED_VIOLET1)
            - LED_V2 (LED_VIOLET2)
        """
        NONE        = 0 
        A           = 1 
        D50         = 2
        D55         = 3
        D65         = 4
        D75         = 5
        D93         = 6
        E           = 7
        F1          = 8
        F2          = 9
        F3          = 10
        F4          = 11
        F5          = 12
        F6          = 13
        F7          = 14
        F8          = 15
        F9          = 16
        F10         = 17
        F11         = 18
        F12         = 19
        LED_B1      = 20
        LED_B2      = 21
        LED_B3      = 22
        LED_B4      = 23
        LED_B5      = 24
        LED_BH1     = 25
        LED_RGB1    = 26
        LED_V1      = 27
        LED_V2      = 28

        INCANDESCENT                    = A
        TUNGSTEN                        = A
        HORIZON                         = D50
        MIDMORNING                      = D55
        DAYLIGHT_NOON                   = D65
        NORTH_SKY                       = D75
        BLUE_PHOSPHOR                   = D93
        EQUAL_ENERGY                    = E
        FLUORESCENT_DAYLIGHT1           = F1
        FLUORESCENT_COOL_WHITE          = F2
        FLUORESCENT_WHITE               = F3
        FLUORESCENT_WARM_WHITE          = F4
        FLUORESCENT_DAYLIGHT2           = F5
        FLUORESCENT_LIGHT_WHITE         = F6
        D65_SIMULATOR                   = F7
        DAYLIGHT_SIMULATOR              = F7
        D50_SIMULATOR                   = F8
        SYLVANIA_F40                    = F8
        FLOURESCENT_COOL_WHITE_DELUXE   = F9
        PHILIPS_TL85                    = F10
        ULTRALUME_50                    = F10
        PHILIPS_TL84                    = F11
        ULTRALUME_40                    = F11
        PHILIPS_TL83                    = F12
        ULTRALUME_30                    = F12
        PHOSPHOR_CONVERTED_BLUE1        = LED_B1
        PHOSPHOR_CONVERTED_BLUE2        = LED_B2
        PHOSPHOR_CONVERTED_BLUE3        = LED_B3
        PHOSPHOR_CONVERTED_BLUE4        = LED_B4
        PHOSPHOR_CONVERTED_BLUE5        = LED_B5
        LED_VIOLET1                     = LED_V1
        LED_VIOLET2                     = LED_V2

    @classmethod
    def wp_from_illuminant(cls, illuminant:Illuminant) -> Float2:
        """
        Look up chromaticity coordinates (white point) of a CIE 1931 2° standard illuminant.

        :param illuminant: Standard illuminant
        :return: White point coordinates (CIE xy)
        """
        # https://en.wikipedia.org/wiki/Standard_illuminant
        ilm = illuminant
        if   ilm == cls.Illuminant.A:        return Float2(0.44757, 0.40745)
        elif ilm == cls.Illuminant.D50:      return Float2(0.34567, 0.35850)
        elif ilm == cls.Illuminant.D55:      return Float2(0.33242, 0.34743)
        elif ilm == cls.Illuminant.D65:      return Float2(0.31271, 0.32902)
        elif ilm == cls.Illuminant.D75:      return Float2(0.29902, 0.31485)
        elif ilm == cls.Illuminant.D93:      return Float2(0.28315, 0.29711)
        elif ilm == cls.Illuminant.E:        return Float2(0.33333, 0.33333)
        elif ilm == cls.Illuminant.F1:       return Float2(0.31310, 0.33727)
        elif ilm == cls.Illuminant.F2:       return Float2(0.37208, 0.37529)
        elif ilm == cls.Illuminant.F3:       return Float2(0.40910, 0.39430)
        elif ilm == cls.Illuminant.F4:       return Float2(0.44018, 0.40329)
        elif ilm == cls.Illuminant.F5:       return Float2(0.31379, 0.34531)
        elif ilm == cls.Illuminant.F6:       return Float2(0.37790, 0.38835)
        elif ilm == cls.Illuminant.F7:       return Float2(0.31292, 0.32933)
        elif ilm == cls.Illuminant.F8:       return Float2(0.34588, 0.35875)
        elif ilm == cls.Illuminant.F9:       return Float2(0.37417, 0.37281)
        elif ilm == cls.Illuminant.F10:      return Float2(0.34609, 0.35986)
        elif ilm == cls.Illuminant.F11:      return Float2(0.38052, 0.37713)
        elif ilm == cls.Illuminant.F12:      return Float2(0.43695, 0.40441)
        elif ilm == cls.Illuminant.LED_B1:   return Float2(0.4560, 0.4078)
        elif ilm == cls.Illuminant.LED_B2:   return Float2(0.4357, 0.4012)
        elif ilm == cls.Illuminant.LED_B3:   return Float2(0.3756, 0.3723)
        elif ilm == cls.Illuminant.LED_B4:   return Float2(0.3422, 0.3502)
        elif ilm == cls.Illuminant.LED_B5:   return Float2(0.3118, 0.3236)
        elif ilm == cls.Illuminant.LED_BH1:  return Float2(0.4474, 0.4066)
        elif ilm == cls.Illuminant.LED_RGB1: return Float2(0.4557, 0.4211)
        elif ilm == cls.Illuminant.LED_V1:   return Float2(0.4560, 0.4548)
        elif ilm == cls.Illuminant.LED_V2:   return Float2(0.3781, 0.3775)
        else: raise Exception(f"Invalid illuminant: {illuminant.name}")

    @staticmethod
    def wp_from_cct(cct:int) -> Float2:
        """
        Compute CIE xy chromaticity coordinates (white point) from correlated color temperature.

        :param cct: Correlated color temperature in range [4000, 25000]
        :return: White point coordinates (CIE xy)
        """    
        # https://github.com/colour-science/colour/blob/develop/colour/temperature/cie_d.py
        # BSD-3-Clause license:

        # Copyright 2013 Colour Developers

        # Redistribution and use in source and binary forms, with or without modification, 
        # are permitted provided that the following conditions are met:

        # 1. Redistributions of source code must retain the above copyright notice, this 
        # list of conditions and the following disclaimer.

        # 2. Redistributions in binary form must reproduce the above copyright notice, 
        # this list of conditions and the following disclaimer in the documentation and/or 
        # other materials provided with the distribution.

        # 3. Neither the name of the copyright holder nor the names of its contributors 
        # may be used to endorse or promote products derived from this software without 
        # specific prior written permission.

        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
        # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
        # IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
        # INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
        # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
        # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
        # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
        # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
        # EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
        assert 4000 <= cct <= 25000, "Correlated color temperature must be in range [4000, 25000]"
        cct = float(cct)

        cct_3 = cct**3
        cct_2 = cct**2
        x = 0.
        if cct <= 7000.:
            x = -4.607 * 10**9 / cct_3 \
            + 2.9678 * 10**6 / cct_2 \
            + 0.09911 * 10**3 / cct \
            + 0.244063
        else:
            x = -2.0064 * 10**9 / cct_3 \
            + 1.9018 * 10**6 / cct_2 \
            + 0.24748 * 10**3 / cct \
            + 0.23704

        y = -3.000 * x**2 + 2.870 * x - 0.275

        return Float2(x, y)

    @staticmethod
    def wp_from_image(im_xyz:Union[torch.Tensor, ColorImage]) -> Float2:
        """
        Estimate the dominant illuminant of an environment map or a target image directly and return its 
        approximate CIE xy chromaticity coordinates (white point).

        .. warning::
         
            This is a lazy method that just averages the pixels in the image tensor. 
            There is no spherical mapping, nor PCA, nor any serious attempt to analyze the image.

        :param im_xyz: Image tensor in CIE XYZ color space
        :type im_xyz: torch.Tensor | ColorImage
        :return: Estimated white point coordinates (CIE xy)
        """
        mean_color = torch.tensor([[[im_xyz[0:1].mean(), im_xyz[1:2].mean(), im_xyz[2:3].mean()]]]).permute(2, 0, 1)
        csum = mean_color[0] + mean_color[1] + mean_color[2]
        mean_color[0] /= csum
        mean_color[1] /= csum
        return Float2(mean_color[0].item(), mean_color[1].item())

    @staticmethod
    def apply(
        im_lms:Union[torch.Tensor, ColorImage], 
        source_white:Union[Float2, Chromaticity, torch.Tensor, numpy.ndarray], 
        target_white:Union[Float2, Chromaticity, torch.Tensor, numpy.ndarray]) -> torch.Tensor:
        """
        Apply white balance.

        :param im_lms: Image tensor in LMS color space
        :type im_lms: torch.Tensor | ColorImage
        :param source_white: Source white point coordinates (CIE xy)
        :type source_white: Float2 | Chromaticity | torch.Tensor | numpy.ndarray
        :param target_white: Target white point coordinates (CIE xy)
        :type target_white: Float2 | Chromaticity | torch.Tensor | numpy.ndarray
        :return: White balanced image tensor
        """
        source = torch.tensor([[[source_white[0], source_white[0], 1.]]], dtype=torch.float32).permute(2, 0, 1)
        target = torch.tensor([[[target_white[0], target_white[0], 1.]]], dtype=torch.float32).permute(2, 0, 1)

        src_lms = ColorSpace.convert(source, ColorSpace.Variant.CIE_XYY, ColorSpace.Variant.LMS)
        dst_lms = ColorSpace.convert(target, ColorSpace.Variant.CIE_XYY, ColorSpace.Variant.LMS)

        mat = [
            [dst_lms[0].item()/src_lms[0].item(), 0., 0.],
            [0., dst_lms[1].item()/src_lms[1].item(), 0.],
            [0., 0., dst_lms[2].item()/src_lms[2].item()]]

        corrected = mm(im_lms, mat)
        return corrected