import typing
import torch
import numpy as np
import io

from .numerics import Float3
from .colorspace import ColorSpace, TransferFunction
from .tonemapping import ToneMapping

class Spectral:
    """
    Spectral color matching functions. Example:

    .. highlight:: python
    .. code-block:: python

        xyz_col = Spectral.wl_to_xyz(550)
    """
    wl_to_xyz_tab = """
        380 0.0014 0.0000 0.0065
        385 0.0022 0.0001 0.0105
        390 0.0042 0.0001 0.0201
        395 0.0076 0.0002 0.0362
        400 0.0143 0.0004 0.0679
        405 0.0232 0.0006 0.1102
        410 0.0435 0.0012 0.2074
        415 0.0776 0.0022 0.3713
        420 0.1344 0.0040 0.6456
        425 0.2148 0.0073 1.0391
        430 0.2839 0.0116 1.3856
        435 0.3285 0.0168 1.6230
        440 0.3483 0.0230 1.7471
        445 0.3481 0.0298 1.7826
        450 0.3362 0.0380 1.7721
        455 0.3187 0.0480 1.7441
        460 0.2908 0.0600 1.6692
        465 0.2511 0.0739 1.5281
        470 0.1954 0.0910 1.2876
        475 0.1421 0.1126 1.0419
        480 0.0956 0.1390 0.8130
        485 0.0580 0.1693 0.6162
        490 0.0320 0.2080 0.4652
        495 0.0147 0.2586 0.3533
        500 0.0049 0.3230 0.2720
        505 0.0024 0.4073 0.2123
        510 0.0093 0.5030 0.1582
        515 0.0291 0.6082 0.1117
        520 0.0633 0.7100 0.0782
        525 0.1096 0.7932 0.0573
        530 0.1655 0.8620 0.0422
        535 0.2257 0.9149 0.0298
        540 0.2904 0.9540 0.0203
        545 0.3597 0.9803 0.0134
        550 0.4334 0.9950 0.0087
        555 0.5121 1.0000 0.0057
        560 0.5945 0.9950 0.0039
        565 0.6784 0.9786 0.0027
        570 0.7621 0.9520 0.0021
        575 0.8425 0.9154 0.0018
        580 0.9163 0.8700 0.0017
        585 0.9786 0.8163 0.0014
        590 1.0263 0.7570 0.0011
        595 1.0567 0.6949 0.0010
        600 1.0622 0.6310 0.0008
        605 1.0456 0.5668 0.0006
        610 1.0026 0.5030 0.0003
        615 0.9384 0.4412 0.0002
        620 0.8544 0.3810 0.0002
        625 0.7514 0.3210 0.0001
        630 0.6424 0.2650 0.0000
        635 0.5419 0.2170 0.0000
        640 0.4479 0.1750 0.0000
        645 0.3608 0.1382 0.0000
        650 0.2835 0.1070 0.0000
        655 0.2187 0.0816 0.0000
        660 0.1649 0.0610 0.0000
        665 0.1212 0.0446 0.0000
        670 0.0874 0.0320 0.0000
        675 0.0636 0.0232 0.0000
        680 0.0468 0.0170 0.0000
        685 0.0329 0.0119 0.0000
        690 0.0227 0.0082 0.0000
        695 0.0158 0.0057 0.0000
        700 0.0114 0.0041 0.0000
        705 0.0081 0.0029 0.0000
        710 0.0058 0.0021 0.0000
        715 0.0041 0.0015 0.0000
        720 0.0029 0.0010 0.0000
        725 0.0020 0.0007 0.0000
        730 0.0014 0.0005 0.0000
        735 0.0010 0.0004 0.0000
        740 0.0007 0.0002 0.0000
        745 0.0005 0.0002 0.0000
        750 0.0003 0.0001 0.0000
        755 0.0002 0.0001 0.0000
        760 0.0002 0.0001 0.0000
        765 0.0001 0.0000 0.0000
        770 0.0001 0.0000 0.0000
        775 0.0001 0.0000 0.0000
        780 0.0000 0.0000 0.0000"""

    @classmethod
    def cm_table(cls, normalize_xyz=False) -> torch.Tensor:
        """
        Returns color matching table as tensor of size [81, 3]. 
        The table contains CIE XYZ values, arranged by wavelength, 
        in increments of 5nm, from 380nm to 780nm.

        :param bool normalize_xyz: Normalize XYZ color, such that X=X/(X+Y+Z), etc
        :return: Color matching function table
        :rtype: torch.Tensor
        """
        f_handler = io.StringIO(cls.wl_to_xyz_tab)
        tab = torch.from_numpy(np.loadtxt(f_handler, usecols=(1,2,3)))
        if normalize_xyz:
            X = tab[...,0:1]
            Y = tab[...,1:2]
            Z = tab[...,2:3]
            x = X / (X + Y + Z)
            y = Y / (X + Y + Z)
            z = Z / (X + Y + Z)
            mask = tab > 0
            tab[mask] = torch.cat([x, y, z], dim=1)[mask]
        f_handler.close()
        return tab

    @classmethod
    def wl_to_xyz(cls, wl:float) -> Float3:
        """
        Wavelength (nm) to CIE XYZ color, linearly interpolated. 
        Precision is limited to interpolated 5nm increments.

        .. note::

            These coordinates will often fall outside the sRGB gamut. 
            Direct color space conversion will yield invalid sRGB values.

        :param float wl: wavelength in nm
        :return: CIE XYZ color
        """
        assert 380 <= wl < 780, "wavelength (nm) must be in range [380, 780]"
        cm_tab = cls.cm_table()
        wl_values = torch.arange(380, 780, 5, dtype=torch.float32)

        # Find the two closest wavelengths in the table
        closest_idx = torch.argmin(torch.abs(wl_values - wl))
        second_closest_idx = closest_idx + 1 if closest_idx < len(wl_values) - 1 else closest_idx - 1

        # Get the XYZ values for the two closest wavelengths
        closest_xyz = cm_tab[closest_idx]
        second_closest_xyz = cm_tab[second_closest_idx]

        # Linear interpolation
        t = (wl - wl_values[closest_idx]) / (wl_values[second_closest_idx] - wl_values[closest_idx])
        interpolated_xyz = closest_xyz + t * (second_closest_xyz - closest_xyz)

        return Float3(interpolated_xyz).clip(0., 1.)

    @classmethod
    def wl_to_srgb_linear(cls, wl:float, normalize:bool=False, lum_scale:float=0.25) -> Float3:
        """
        Wavelength (nm) to normalized, approximate linear sRGB color, clamped to [0, 1] range.

        :param wl: wavelength in nm
        :param normalize: normalize sRGB color
        :return: sRGB linear color
        """
        xyz = cls.wl_to_xyz(wl)
        cs = ColorSpace.Variant
        xyy = ColorSpace.convert(torch.from_numpy(xyz[..., np.newaxis, np.newaxis]), cs.CIE_XYZ, cs.CIE_XYY)
        xyy[2] *= lum_scale
        srgb = ColorSpace.convert(xyy, cs.CIE_XYY, cs.SRGB_LIN)
        srgb = Float3(srgb)
        if normalize:
            if np.any(srgb < 0.): srgb -= srgb.min()
            if srgb.sum() > 0.: srgb /= srgb.max()

        return srgb #.clip( 0., 1.)

    @classmethod
    def wl_to_srgb(cls, wl:float, normalize:bool=False, lum_scale:float=0.25) -> Float3:
        """
        Wavelength (nm) to normalized, approximate sRGB color, 
        clamped to [0, 1] range, with sRGB gamma curve.

        .. note::
        
            Wolfram Alpha doesn't quite agree, but it's rather close. This produces a 
            plausible-looking spectrum, but there is probably some missing (pre?) normalization step. 
            Take the RGB outputs with a grain of salt.

        :param wl: wavelength in nm
        :param normalize: normalize sRGB color
        :return: sRGB linear color
        """
        lin = cls.wl_to_srgb_linear(wl, normalize=normalize, lum_scale=lum_scale)
        ten = torch.from_numpy(lin[..., np.newaxis, np.newaxis])
        return Float3(TransferFunction.srgb_oetf(ten))