from __future__ import annotations
import typing
from typing import Union
import torch
import numpy as np
from enum import IntEnum

from .numerics import Float2, matmul_tl as mm

class ColorSpace:
    """
    Color space conversion. Applies OETFs and EOTFs as needed but omits tonemapping. Cylindrical transformations are 
    treated as distinct color spaces. Example:

    .. highlight:: python
    .. code-block:: python
        
        cs_in = ColorSpace.Variant.SRGB_LIN
        cs_out = ColorSpace.Variant.OKLAB
        oklab_image = ColorSpace.convert(srgb_image, source=cs_in, destination=cs_out)
    """
    class Variant(IntEnum):
        """
        Color space enum. For a list of available options, see :ref:`ref_color_spaces`.
        """
        UNKNOWN     = 1<<0  
        NONCOLOR    = 1<<1  
        CIE_XYZ     = 1<<2  
        CIE_XYY     = 1<<3  
        SRGB        = 1<<4  
        SRGB_LIN    = 1<<5  
        REC709      = 1<<6  
        REC2020     = 1<<7  
        REC2020_LIN = 1<<8  
        DCI_P3      = 1<<9  
        DCI_P3_LIN  = 1<<10 
        DISPLAY_P3  = 1<<11 
        ACESCG      = 1<<12 
        ACESCC      = 1<<13 
        ACESCCT     = 1<<14 
        ACES2065_1  = 1<<15 
        LMS         = 1<<16 
        OKLAB       = 1<<17 
        CIELAB      = 1<<18 
        CIELUV      = 1<<19 
        HSV         = 1<<20 
        HSL         = 1<<21 
        OKHSV       = 1<<22
        OKHSL       = 1<<23

        SCENE_LINEAR    = SRGB_LIN | REC2020_LIN | DCI_P3_LIN | ACESCG | ACES2065_1 | CIE_XYZ
        PERCEPTUAL      = OKLAB | CIELAB | CIELUV | OKHSL | OKHSV
        CYLINDRICAL     = HSL | HSV | OKHSL | OKHSV

        GAMUT_SRGB      = SRGB | SRGB_LIN | REC709 | HSL | HSV
        GAMUT_AP0       = ACES2065_1
        GAMUT_AP1       = ACESCG | ACESCC | ACESCCT
        GAMUT_REC2020   = REC2020 | REC2020_LIN
        GAMUT_DCI_P3    = DCI_P3 | DCI_P3_LIN
        GAMUT_DISPLAY_P3= DISPLAY_P3
        GAMUT_OKLAB     = OKLAB | OKHSL | OKHSV
        GAMUT_CIE_XYZ   = CIE_XYZ | CIE_XYY
        GAMUT_CIELAB    = CIELAB
        GAMUT_CIELUV    = CIELUV
        GAMUT_OTHER     = LMS | UNKNOWN | NONCOLOR

        WP_D65          = SRGB | SRGB_LIN | REC709 | DISPLAY_P3 | REC2020 | REC2020_LIN | CIE_XYZ | CIE_XYY
        WP_CCT_6300     = DCI_P3 | DCI_P3_LIN
        WP_CCT_6000     = ACESCG | ACESCC | ACESCCT | ACES2065_1

        MODEL_RGB       = SRGB | SRGB_LIN | REC709 | REC2020 | REC2020_LIN | DCI_P3 | DCI_P3_LIN | DISPLAY_P3 | \
                        ACESCG | ACESCC | ACESCCT | ACES2065_1
        MODEL_CIE       = CIE_XYZ | CIE_XYY | CIELAB | CIELUV
        MODEL_CAM       = 0
        MODEL_YUV       = 0
        MODEL_OTHER     = LMS | HSL | HSV | OKLAB # is OKLAB CAM-based?
        
        NEGATIVE        = OKLAB | CIELAB | CIELUV | GAMUT_AP0
        NON_NEGATIVE    = ~NEGATIVE

        DISABLED        = CIELUV
        UNSUPPORTED     = OKHSV | OKHSL # disabled doesn't go here - CS must have alternate path
        SUPPORTED       = ~UNSUPPORTED 

        # FIXME: LUV doesn't quite match expected values, needs further testing

    mat_xyz_to_srgb = [
        [3.24096994190452134, -1.53738317757009346, -0.498610760293003284],
        [-0.969243636280879826, 1.87596750150772067, 0.0415550574071756125],
        [0.0556300796969936084, -0.203976958888976564, 1.05697151424287856]]

    mat_srgb_to_xyz = [
        [0.412390799265959481, 0.357584339383877964, 0.180480788401834288],
        [0.212639005871510358, 0.715168678767755927, 0.072192315360733715],
        [0.0193308187155918507, 0.119194779794625988, 0.950532152249660581]]

    mat_srgb_to_acescg = [
        [ 0.6130974024, 0.3395231462, 0.04737945141],
        [ 0.07019372247, 0.916353879, 0.01345239847],
        [ 0.02061559288, 0.1095697729, 0.8698146341]]

    # NOTE: Includes "D60"/D65 white point conversion
    mat_acescg_to_srgb = [
        [ 1.705050993,  -0.6217921206,-0.083258872],
        [-0.1302564175,  1.140804737, -0.01054831907],
        [-0.02400335681,-0.1289689761, 1.152972333]]

    # NOTE: Includes "D60"/D65 white point conversion
    mat_srgb_to_aces2065_1 = [
        [ 0.439632982,  0.382988698, 0.17737832],
        [ 0.0897764431, 0.813439429, 0.0967841284],
        [ 0.0175411704, 0.111546553, 0.870912277]]

    mat_aces2065_1_to_srgb = [
        [ 2.52168619,  -1.13413099, -0.387555198],
        [-0.276479914,  1.37271909, -0.0962391736],
        [-0.015378065, -0.152975336, 1.1683534]]

    mat_srgb_to_displayp3 = [
        [ 0.822461969,  0.177538031,  1.15772692e-10],
        [ 0.0331941989, 0.966805801,  1.95085037e-11],
        [ 0.0170826307, 0.0723974405, 0.910519929]]

    mat_displayp3_to_srgb = [
        [ 1.22494018,  -0.224940176, -4.77534979e-11],
        [-0.0420569547, 1.04205695,   3.37864801e-11],
        [-0.0196375546,-0.0786360454, 1.0982736]] 

    # NOTE: No chromatic adaptation
    mat_srgb_to_dcip3 = [
        [0.868579739716132409,  0.128919138460847047,  0.00250112182302054368],
        [0.0345404102543194426, 0.961811386361919975,  0.0036482033837605824],
        [0.0167714290414502718, 0.0710399977868858352, 0.912188573171663893]]

    # NOTE: No chromatic adaptation
    mat_dcip3_to_srgb = [
        [ 1.15751640619975871,  -0.154962378073857756, -0.00255402812590095854],
        [-0.0415000715306859699, 1.04556792307969925,  -0.00406785154901328463],
        [-0.0180500389562539583,-0.0785782726530290654, 1.09662831160928302]]

    # NOTE: No chromatic adaptation
    mat_dcip3_to_xyz = [
        [ 0.445169815564552417,    0.277134409206777664,  0.172282669815564564],
        [ 0.209491677912730539,    0.721595254161043636,  0.0689130679262258258],
        [-3.63410131696985616e-17, 0.0470605600539811521, 0.907355394361973415]]

    # NOTE: No chromatic adaptation
    mat_xyz_to_dcip3 = [
        [2.7253940304917328, -1.01800300622718496, -0.440163195190036463],
        [-0.795168025808764195, 1.689732054843624, 0.0226471906084774533],
        [0.0412418913957000325, -0.0876390192158623825, 1.10092937864632191]]

    mat_srgb_to_rec2020 = [
        [ 0.627403896,  0.329283039,  0.0433130657],
        [ 0.0690972894, 0.919540395,  0.0113623156],
        [ 0.0163914389, 0.0880133077, 0.895595253]]

    mat_rec2020_to_srgb = [
        [ 1.660491,    -0.587641139,-0.0728498633],
        [-0.124550475, 1.1328999,   -0.00834942258],
        [-0.0181507633,-0.100578898, 1.11872966]]

    mat_rec2020_to_xyz = [
        [0.636958048301291, 0.144616903586208, 0.168880975164172],
        [0.262700212011267, 0.677998071518871, 0.059301716469862],
        [4.99410657446607e-17, 0.0280726930490874, 1.06098505771079]]

    mat_xyz_to_rec2020 = [
        [1.71665118797127, -0.355670783776393, -0.25336628137366],
        [-0.666684351832489, 1.61648123663494, 0.0157685458139111],
        [0.0176398574453108, -0.0427706132578085, 0.942103121235474]]

    # NOTE: No chromatic adaptation
    mat_acescg_to_xyz = [
        [ 0.66245418, 0.13400421, 0.15618769],
        [ 0.27222872, 0.67408177, 0.05368952],
        [-0.00557465, 0.00406073, 1.0103391 ]]

    # NOTE: No chromatic adaptation
    mat_xyz_to_acescg = [
        [ 1.64102338, -0.32480329, -0.2364247 ],
        [-0.66366286,  1.61533159,  0.01675635],
        [ 0.01172189, -0.00828444,  0.98839486]]

    # NOTE: For CIE XYZ color
    mat_d60_to_d65 = [
        [ 0.98722400,-0.00611327, 0.01595330],
        [-0.00759836, 1.00186000, 0.00533002],
        [ 0.00307257,-0.00509595, 1.08168000]]

    # NOTE: For CIE XYZ color
    mat_d65_to_d60 = [
        [ 1.01303000, 0.00610531,-0.01497100],
        [ 0.00769823, 0.99816500,-0.00503203],
        [-0.00284131, 0.00468516, 0.92450700]]

    # NOTE: For CIE XYZ color
    mat_d65_to_dci = [
        [0.976578896646979768, -0.0154362646984919742, -0.016686021704209866],
        [-0.0256896658505145926, 1.02853916787996963, -0.00378517365630504153],
        [-0.00570574587417104179, 0.0110778657389971485, 0.871176159390377409]]
    
    # NOTE: For CIE XYZ color
    mat_dci_to_d65 = [
        [1.02449672775257752, 0.0151635410224165156, 0.0196885223342066827],
        [0.0256121933371584198, 0.97258630562441342, 0.00471635229242730096],
        [0.0063842306500876874, -0.012268082736730219, 1.14794244517367791]]

    mat_xyz_to_lms = [
        [ 0.8951, 0.2664,-0.1614],
        [-0.7502, 1.7135, 0.0367],
        [ 0.0389,-0.0685, 1.0296]]

    mat_lms_to_xyz = [
        [ 0.986993,   -0.147054,  0.159963],
        [ 0.432305,    0.51836,   0.0492912],
        [ -0.00852866, 0.0400428, 0.968487]]

    # OKLAB's XYZ to LMS
    mat_oklab_m1 = [
        [ 0.8189330101,  0.3618667424, -0.1288597137],
        [ 0.0329845436,  0.9293118715,  0.0361456387],
        [ 0.0482003018,  0.2643662691,  0.6338517070]]

    # OKLAB's non-linear L'M'S' to OKLAB
    mat_oklab_m2 = [
        [ 0.2104542553,  0.7936177850, -0.0040720468],
        [ 1.9779984951, -2.4285922050,  0.4505937099],
        [ 0.0259040371,  0.7827717662, -0.8086757660]]

    # Inverse of OKLAB M1
    mat_oklab_m1_inv = [
        [ 1.22701385, -0.55779998,  0.28125615],
        [-0.04058018,  1.11225687, -0.07167668],
        [-0.07638128, -0.42148198,  1.58616322]]

    # Inverse of OKLAB M2
    mat_oklab_m2_inv = [
        [ 1.        ,  0.39633779,  0.21580376],
        [ 1.00000001, -0.10556134, -0.06385417],
        [ 1.00000005, -0.08948418, -1.29148554]]

    @classmethod
    def convert(cls, im:Union[torch.Tensor, ColorImage], source:Variant, destination:Variant) -> torch.Tensor:
        """
        Change the color space of an image. Cylindrical transformations HSV/HSL are 
        treated as their own color spaces and assumed to be relative to sRGB linear. 
        Unless otherwise noted or required by specification (e.g. ACES), we assume D65 white point.

        .. warning::

            Tone mapping is not included, so converting the color space of HDR values to  
            an LDR-designated color space will not automatically reduce dynamic range. For example, 
            taking an HDR image from :code:`ACESCG` (AP1) to :code:`SRGB` will yield the sRGB 
            gamma curve, but values outside the required range must still be tone mapped or clamped beforehand.

        .. warning::

            Cylindrical transformations (HSL, HSV) should be given input in [0, 1] linear sRGB range 
            (or equivalent). This is not strictly enforced but input outside this range may yield 
            unpredictable results or *NaN* values.

        :param im: [C=3, H, W] image tensor 
        :type im: torch.Tensor | ColorImage
        :param source: color space to convert from
        :param destination: color space to convert to
        :return: image tensor in designated color space
        """
        ip, op = source, destination
        cs = cls.Variant
        tf = TransferFunction
        if ip == op: return im

        assert im.dim() == 3 and im.size(0) == 3, f"expected [C=3, H, W] image tensor, got {im.size()}"
        assert source != 0,             f"Unknown source color space"
        assert ip & cs.SUPPORTED,       f"Source color space not supported: {source.name}"
        assert op & cs.SUPPORTED,       f"Destination color space not supported: {destination.name}"
        assert ip & ~cs.DISABLED,       f"Source color space disabled: {ColorSpace.Variant(ip).name}"
        assert op & ~cs.DISABLED,       f"Destination color space disabled: {ColorSpace.Variant(op).name}"

        err_not_implemented = f"Color space conversion not implemented: {ColorSpace.Variant(ip).name} to {ColorSpace.Variant(op).name}" 

        # Direct path where it matters, loop-de-loop elsewhere
        if ip == cs.SRGB_LIN:
            if   op == cs.SRGB:         im = tf.srgb_oetf(im)
            elif op == cs.REC709:       im = tf.rec709_oetf(im)
            elif op == cs.REC2020:      im = tf.rec2020_oetf(mm(im, cls.mat_srgb_to_rec2020))
            elif op == cs.REC2020_LIN:  im = mm(im, cls.mat_srgb_to_rec2020)
            elif op == cs.DCI_P3:       im = tf.dcip3_oetf(mm(mm(mm(im, cls.mat_srgb_to_xyz), cls.mat_d65_to_dci), cls.mat_xyz_to_dcip3))
            elif op == cs.DCI_P3_LIN:   im = mm(mm(mm(im, cls.mat_srgb_to_xyz), cls.mat_d65_to_dci), cls.mat_xyz_to_dcip3)
            elif op == cs.DISPLAY_P3:   im = tf.srgb_oetf(mm(im, cls.mat_srgb_to_displayp3))
            elif op == cs.CIE_XYZ:      im = mm(im, cls.mat_srgb_to_xyz)
            elif op == cs.CIE_XYY:      im = cls._xyz_to_xyy(mm(im, cls.mat_srgb_to_xyz))
            elif op == cs.LMS:          im = cls._xyz_to_lms(mm(im, cls.mat_srgb_to_xyz))
            elif op == cs.ACESCG:       im = mm(im, cls.mat_srgb_to_acescg)
            elif op == cs.ACESCC:       im = cls._acescg_to_acescc(mm(im, cls.mat_srgb_to_acescg))
            elif op == cs.ACES2065_1:   im = mm(im, cls.mat_srgb_to_aces2065_1)
            elif op == cs.CIELAB:       im = cls._xyz_to_cielab(mm(im, cls.mat_srgb_to_xyz))
            elif op == cs.CIELUV:       im = cls._xyz_to_cieluv(mm(im, cls.mat_srgb_to_xyz))
            elif op == cs.OKLAB:        im = cls._rgb_to_oklab(im)
            elif op == cs.HSL:          im = cls._rgb_to_hsl(tf.srgb_oetf(im))
            elif op == cs.HSV:          im = cls._rgb_to_hsv(tf.srgb_oetf(im))
            else:                       raise Exception(err_not_implemented)
        elif ip == cs.SRGB:
            if   op == cs.HSL:          im = cls._rgb_to_hsl(im)
            elif op == cs.HSV:          im = cls._rgb_to_hsv(im)
            else:                       im = cls.convert(tf.srgb_eotf(im), cs.SRGB_LIN, op)
        elif ip == cs.REC709:           im = cls.convert(tf.rec709_eotf(im), cs.SRGB_LIN, op)
        elif ip == cs.REC2020:          
            if op == cs.REC2020_LIN:    im = tf.rec2020_eotf(im)
            elif op == cs.CIE_XYZ:      im = mm(tf.rec2020_eotf(im), cls.mat_rec2020_to_xyz)
            elif op == cs.SRGB_LIN:     im = mm(tf.rec2020_eotf(im), cls.mat_rec2020_to_srgb)
            else:                       im = cls.convert(mm(tf.rec2020_eotf(im), cls.mat_rec2020_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.REC2020_LIN:      
            if op == cs.REC2020:        im = tf.rec2020_oetf(im)
            elif op == cs.CIE_XYZ:      im = mm(im, cls.mat_rec2020_to_xyz)
            elif op == cs.SRGB_LIN:     im = mm(im, cls.mat_rec2020_to_srgb)
            else:                       im = cls.convert(mm(im, cls.mat_rec2020_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.DCI_P3:           
            if op == cs.DCI_P3_LIN:     im = tf.dcip3_eotf(im)
            elif op == cs.CIE_XYZ:      im = mm(mm(tf.dcip3_eotf(im), cls.mat_dcip3_to_xyz), cls.mat_dci_to_d65)
            else:                       im = cls.convert(mm(mm(tf.dcip3_eotf(im), cls.mat_dcip3_to_xyz), cls.mat_dci_to_d65), cs.CIE_XYZ, op)
        elif ip == cs.DCI_P3_LIN:       
            if op == cs.DCI_P3:         im = tf.dcip3_oetf(im)
            elif op == cs.CIE_XYZ:      im = mm(mm(im, cls.mat_dcip3_to_xyz), cls.mat_dci_to_d65)
            else:                       im = cls.convert(mm(mm(im, cls.mat_dcip3_to_xyz), cls.mat_dci_to_d65), cs.CIE_XYZ, op)
        elif ip == cs.DISPLAY_P3:       im = cls.convert(mm(tf.srgb_eotf(im), cls.mat_displayp3_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.CIE_XYZ:
            if   op == cs.CIE_XYY:      im = cls._xyz_to_xyy(im)
            elif op == cs.REC2020_LIN:  im = mm(im, cls.mat_xyz_to_rec2020)
            elif op == cs.REC2020:      im = tf.rec2020_oetf(mm(im, cls.mat_xyz_to_rec2020))
            elif op == cs.DCI_P3_LIN:   im = mm(mm(im, cls.mat_d65_to_dci), cls.mat_xyz_to_dcip3)
            elif op == cs.DCI_P3:       im = tf.dcip3_oetf(mm(mm(im, cls.mat_d65_to_dci), cls.mat_xyz_to_dcip3))
            elif op == cs.LMS:          im = cls._xyz_to_lms(im)
            elif op == cs.ACESCG:       im = mm(cls._d65_to_d60(im), cls.mat_xyz_to_acescg)
            elif op == cs.CIELAB:       im = cls._xyz_to_cielab(im)
            elif op == cs.CIELUV:       im = cls._xyz_to_cieluv(im)
            elif op == cs.OKLAB:        im = cls._xyz_to_oklab(im)
            else:                       im = cls.convert(mm(im, cls.mat_xyz_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.CIE_XYY:          
            if   op == cs.CIE_XYZ:      im = cls._xyy_to_xyz(im)
            else:                       im = cls.convert(cls._xyy_to_xyz(im), cs.CIE_XYZ, op)
        elif ip == cs.LMS:              
            if   op == cs.CIE_XYZ:      im = cls._lms_to_xyz(im)
            else:                       im = cls.convert(cls._lms_to_xyz(im), cs.CIE_XYZ, op)
        elif ip == cs.ACESCG:
            # if   op == cs.CIE_XYZ:      im = cls._d60_to_d65(mm(im, cls.mat_acescg_to_xyz)) # FIXME: fails unit test (?)
            if   op == cs.ACESCC:       im = cls._acescg_to_acescc(im)
            else:                       im = cls.convert(mm(im, cls.mat_acescg_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.ACESCC:
            if   op == cs.ACESCG:       im = cls._acescc_to_acescg(im)
            else:                       im = cls.convert(cls._acescc_to_acescg(im), cs.ACESCG, op)
        elif ip == cs.ACES2065_1:       im = cls.convert(mm(im, cls.mat_aces2065_1_to_srgb), cs.SRGB_LIN, op)
        elif ip == cs.HSL:
            if   op == cs.SRGB:         im = cls._hsl_to_rgb(im)
            else:                       im = cls.convert(tf.srgb_eotf(cls._hsl_to_rgb(im)), cs.SRGB_LIN, op)
        elif ip == cs.HSV:
            if   op == cs.SRGB:         im = cls._hsv_to_rgb(im)
            else:                       im = cls.convert(tf.srgb_eotf(cls._hsv_to_rgb(im)), cs.SRGB_LIN, op)
        elif ip == cs.CIELAB:           im = cls.convert(cls._cielab_to_xyz(im), cs.CIE_XYZ, op)
        elif ip == cs.CIELUV:           im = cls.convert(cls._cieluv_to_xyz(im), cs.CIE_XYZ, op)
        elif ip == cs.OKLAB:
            if   op == cs.CIE_XYZ:      im = cls._oklab_to_xyz(im)
            else: im = cls.convert(cls._oklab_to_rgb(im), cs.SRGB_LIN, op)
        else: raise Exception(err_not_implemented)

        return im

    @classmethod
    def _xyz_to_xyy(cls, xyz:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE XYZ color space to CIE xyY color space.

        :param xyz: Input CIE XYZ color space tensor
        :return: CIE xyY color space tensor
        """
        X = xyz[0:1]
        Y = xyz[1:2]
        Z = xyz[2:3]
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        return torch.cat([x, y, Y], dim=0)

    @classmethod
    def _xyy_to_xyz(cls, xyy:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE xyY color space to CIE XYZ color space.

        :param xyy: Input CIE xyY color space tensor
        :return: CIE XYZ color space tensor
        """
        x = xyy[0:1]
        y = xyy[1:2]
        Y = xyy[2:3]
        X = (Y / y) * x
        Z = (Y / y) * (1. - x - y)
        return torch.cat([X, Y, Z], dim=0)

    @classmethod
    def _xyz_to_lms(cls, xyz:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE XYZ color space to LMS color space.

        :param xyz: Input CIE XYZ color space tensor
        :return: LMS color space tensor
        """
        return mm(xyz, cls.mat_xyz_to_lms)

    @classmethod
    def _lms_to_xyz(cls, lms:torch.Tensor) -> torch.Tensor:
        """
        Convert LMS color space to CIE XYZ color space.

        :param lms: Input LMS color space tensor
        :return: CIE XYZ color space tensor
        """
        return mm(lms, cls.mat_lms_to_xyz)

    @classmethod
    def _acescg_to_acescc(cls, cg:torch.Tensor) -> torch.Tensor:
        """
        Convert scene-linear ACEScg to log ACEScc.

        :param lms: Input ACEScg color space tensor
        :return: ACEScc color space tensor
        """
        res = torch.where(cg < 0.00003051757, 
            (torch.log2(0.00001525878 + cg * 0.5) + 9.72) / 17.52, 
            (torch.log2(cg) + 9.72) / 17.52)
        return res

    @classmethod
    def _acescc_to_acescg(cls, cc:torch.Tensor) -> torch.Tensor:
        """
        Convert log ACEScc to scene-linear ACEScg.

        :param lms: Input ACEScc color space tensor
        :return: ACEScg color space tensor
        """
        res = torch.where(cc < -0.3013698630, 
            (torch.exp2(cc * 17.52 - 9.72) - 0.00001525878) * 2,
            torch.exp2(cc * 17.52 - 9.72))
        return res

    @classmethod
    def _xyz_to_oklab(cls, xyz:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE XYZ color space to OKLAB color space.

        :param xyz: Input CIE XYZ color space tensor
        :return: OKLAB color space tensor
        """      
        lms = mm(xyz, cls.mat_oklab_m1)
        lms_p = torch.pow(torch.abs(lms), 0.3333333333) * torch.sign(lms).float()
        lab = mm(lms_p, cls.mat_oklab_m2)
        return lab

    @classmethod
    def _oklab_to_xyz(cls, lab:torch.Tensor) -> torch.Tensor:
        """
        Convert OKLAB color space to CIE XYZ color space.

        :param lab: Input OKLAB color space tensor
        :return: CIE XYZ color space tensor
        """
        lms_p = mm(lab, cls.mat_oklab_m2_inv)
        lms = torch.pow(lms_p, 3.)
        xyz = mm(lms, cls.mat_oklab_m1_inv)
        return xyz


    @classmethod
    def __pivot_xyz_to_lab(cls, val):        
        return torch.where(val > 0.008856, torch.pow(val, 0.3333333333), ((val * 903.3) + 16.0) / 116.0)

    @classmethod
    def _xyz_to_cielab(cls, xyz:torch.Tensor) -> torch.Tensor:
        """
        Convert color space from CIE XYZ to CIELAB.

        :param xyz: Input CIE XYZ color space tensor
        :return: CIELAB color space tensor
        """
        # https://github.com/CairX/convert-colors-py/blob/master/convcolors/__init__.py
        # MIT License

        # Copyright (c) 2022 Thomas Cairns

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.        
        x = xyz[0:1] / 0.95047 
        y = xyz[1:2] / 1.00000 
        z = xyz[2:3] / 1.08883 

        x = cls.__pivot_xyz_to_lab(x)
        y = cls.__pivot_xyz_to_lab(y)
        z = cls.__pivot_xyz_to_lab(z)

        l = torch.maximum(torch.zeros_like(y).to(y.device), (116.0 * y) - 16.0)
        a = (x - y) * 500.0
        b = (y - z) * 200.0
        return torch.cat([l, a, b], dim=0)

    @classmethod
    def _cielab_to_xyz(cls, lab:torch.Tensor) -> torch.Tensor:
        """
        Convert color space from CIELAB to CIE XYZ.
        
        .. note::

            Assumes D65 standard illuminant.

        :param lab: Input CIELAB color space tensor
        :return: CIE XYZ color space tensor
        """
        # https://github.com/CairX/convert-colors-py/blob/master/convcolors/__init__.py
        # MIT License

        # Copyright (c) 2022 Thomas Cairns

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        l = lab[0:1]
        a = lab[1:2]
        b = lab[2:3]

        # Reminder: The y values is calculated first as it can be reused
        # for the calculation of x and z.
        y = (l + 16.0) / 116.0
        x = y + (a / 500.0)
        z = y - (b / 200.0)

        x3 = x * x * x
        z3 = z * z * z
        y3 = y * y * y

        x = torch.where(x3 > 0.008856, x3, ((x * 116.0) - 16.0) / 903.3)
        y = torch.where(l > 7.9996248, y3, l / 903.3)
        z = torch.where(z3 > 0.008856, z3, ((z * 116.0) - 16.0) / 903.3)

        x = x * 0.95047 
        y = y * 1.00000 
        z = z * 1.08883

        return torch.cat([x, y, z], dim=0)

    def _xyz_to_cieluv(image:torch.Tensor) -> torch.Tensor:
        """
        Converts CIE XYZ to CIELUV. 
        
        .. note::

            Assumes D65 standard illuminant.

        :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are X, Y, Z
        :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are L, U, V
        """
        # https://github.com/stefanLeong/S2CRNet/blob/main/scripts/utils/color.py
        # MIT License

        # Copyright (c) 2021 StefanLeong

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        if len(image.size()) == 3:
            small_L = (29. / 3) ** 3 * image[1]
            large_L = 116 * torch.pow(image[1], 1 / 3.) - 16
            L = torch.where(image[1] <= (6. / 29) ** 3, small_L, large_L)

            denom = (image[0] + 15 * image[1] + 3 * image[2])
            u_prime = torch.where(denom != 0., 4 * image[0] / denom, 0.)
            v_prime = torch.where(denom != 0., 9 * image[1] / denom, 0.)
            d = 0
        elif len(image.size()) == 4:
            small_L = (29. / 3) ** 3 * image[:, 1]
            large_L = 116 * torch.pow(image[:, 1], 1 / 3.) - 16
            L = torch.where(image[:, 1] <= (6. / 29) ** 3, small_L, large_L)

            denom = (image[:, 0] + 15 * image[:, 1] + 3 * image[:, 2])
            u_prime = torch.where(denom > 0., 4 * image[:, 0] / denom, 0.)
            v_prime = torch.where(denom > 0., 9 * image[:, 1] / denom, 0.)
            d = 1

        u = 13 * L * (u_prime - .2009)
        v = 13 * L * (v_prime - .4610)

        luv_image = torch.stack((L, u, v), dim=d)

        return luv_image

    def _cieluv_to_xyz(image:torch.Tensor) -> torch.Tensor:
        """
        Converts CIELUV to CIE XYZ. 
        
        .. note::

            Assumes D65 standard illuminant.

        :param image: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are L, U, V
        :return: A pytorch tensor of shape (3, n_pixels_x, n_pixels_y) in which the channels are X, Y, Z
        """
        # https://github.com/stefanLeong/S2CRNet/blob/main/scripts/utils/color.py
        # MIT License

        # Copyright (c) 2021 StefanLeong

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        if len(image.size()) == 3:
            denom = (13 * image[0])
            u_prime = torch.where(denom != 0., image[1] / denom, 0.) + .2009
            v_prime = torch.where(denom != 0., image[2] / denom, 0.) + .4610

            small_Y = image[0] * (3. / 29) ** 3
            large_Y = ((image[0] + 16.) / 116.) ** 3

            Y = torch.where(image[0] <= 8, small_Y, large_Y)
            d = 0
            # batch of images
        elif len(image.size()) == 4:
            denom = (13 * image[:, 0])
            u_prime = torch.where(denom != 0., image[:, 1] / denom, 0.) + .2009
            v_prime = torch.where(denom != 0., image[:, 2] / denom, 0.) + .4610

            small_Y = image[:, 0] * (3. / 29) ** 3
            large_Y = ((image[:, 0] + 16.) / 116.) ** 3

            Y = torch.where(image[:, 0] <= 8, small_Y, large_Y)
            d = 1

        X = torch.where(v_prime != 0., Y * 9 * u_prime / (4 * v_prime), 0.)
        Z = torch.where(v_prime != 0., Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime), 0.)

        xyz_image = torch.stack((X, Y, Z), dim=d)

        return xyz_image

    @classmethod
    def _rgb_to_oklab(cls, rgb:torch.Tensor) -> torch.Tensor:
        """
        Convert color space from linear sRGB to OKLAB.

        :param rgb: Input linear sRGB color space tensor
        :return: OKLAB color space tensor
        """
        cr = rgb[0:1]
        cg = rgb[1:2]
        cb = rgb[2:3]

        l = 0.4122214708 * cr + 0.5363325363 * cg + 0.0514459929 * cb;
        m = 0.2119034982 * cr + 0.6806995451 * cg + 0.1073969566 * cb;
        s = 0.0883024619 * cr + 0.2817188376 * cg + 0.6299787005 * cb;

        l_ = torch.pow(torch.abs(l), 0.3333333333) * torch.sign(l).float()
        m_ = torch.pow(torch.abs(m), 0.3333333333) * torch.sign(m).float()
        s_ = torch.pow(torch.abs(s), 0.3333333333) * torch.sign(s).float()

        return torch.cat([
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_], dim=0)

    @classmethod
    def _oklab_to_rgb(cls, lab:torch.Tensor) -> torch.Tensor:
        """
        Convert color space from OKLAB to linear sRGB.

        :param lab: Input OKLAB color space tensor
        :return: Linear sRGB color space tensor
        """
        cl = lab[0:1]
        ca = lab[1:2]
        cb = lab[2:3]

        l_ = cl + 0.3963377774 * ca + 0.2158037573 * cb
        m_ = cl - 0.1055613458 * ca - 0.0638541728 * cb
        s_ = cl - 0.0894841775 * ca - 1.2914855480 * cb

        l = l_*l_*l_
        m = m_*m_*m_
        s = s_*s_*s_

        return torch.cat([
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], dim=0)

    @classmethod
    def _rgb_to_hsl(cls, rgb: torch.Tensor) -> torch.Tensor:
        """
        Transform sRGB image tensor to sRGB-relative HSL. 
        
        .. note::

            expects non-linear sRGB w/ gamma curve as input

        :param rgb: Input sRGB image tensor
        :return: HSL image tensor
        """
        # https://github.com/windingwind/seal-3d/blob/main/SealNeRF/color_utils.py
        # MIT License

        # Copyright (c) 2022 hawkey

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        rgb = rgb.clone().unsqueeze(0)
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsl_h[cmax_idx == 3] = 0.
        hsl_h /= 6.

        hsl_l = (cmax + cmin) / 2.
        hsl_s = torch.empty_like(hsl_h)
        hsl_s[hsl_l == 0] = 0
        hsl_s[hsl_l == 1] = 0
        hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
        hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
        hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
        hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
        hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
        return torch.cat([hsl_h, hsl_s, hsl_l], dim=1).squeeze(0)

    @classmethod
    def _hsl_to_rgb(cls, hsl: torch.Tensor) -> torch.Tensor:
        """
        Transform sRGB-relative HSL image tensor to sRGB. 
        
        .. note::

            returns non-linear sRGB w/ gamma curve as output

        :param hsl: Input HSL image tensor
        :return: sRGB image tensor
        """
        # https://github.com/windingwind/seal-3d/blob/main/SealNeRF/color_utils.py
        # MIT License

        # Copyright (c) 2022 hawkey

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        hsl = hsl.clone().unsqueeze(0)
        hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
        _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
        _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
        _m = hsl_l - _c / 2.
        idx = (hsl_h * 6.).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsl).to(hsl.device)
        _o = torch.zeros_like(_c).to(hsl.device)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb.squeeze(0)

    @classmethod
    def _rgb_to_hsv(cls, rgb: torch.Tensor) -> torch.Tensor:
        """
        Transform sRGB image tensor to sRGB-relative HSV. 
        
        .. note::

            expects non-linear sRGB w/ gamma curve as input

        .. warning::

            input tensor will be clamped to [0, 1] range

        :param rgb: Input sRGB image tensor
        :return: HSV image tensor
        """
        # https://github.com/windingwind/seal-3d/blob/main/SealNeRF/color_utils.py
        # MIT License

        # Copyright (c) 2022 hawkey

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        rgb = rgb.clone().clamp(0.,1.).unsqueeze(0)
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1).squeeze(0)

    @classmethod
    def _hsv_to_rgb(cls, hsv: torch.Tensor) -> torch.Tensor:
        """
        Transform sRGB-relative HSV image tensor to sRGB. 
        
        .. note::
        
            returns non-linear sRGB w/ gamma curve as output

        :param hsv: Input HSV image tensor
        :return: sRGB image tensor
        """
        # https://github.com/windingwind/seal-3d/blob/main/SealNeRF/color_utils.py
        # MIT License

        # Copyright (c) 2022 hawkey

        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:

        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.

        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
        hsv = hsv.clone().unsqueeze(0)
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c).to(hsv.device)
        idx = (hsv_h * 6.).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv).to(hsv.device)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb.squeeze(0)

    @classmethod
    def _d60_to_d65(cls, im:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE XYZ image from "D60" to D65 white point.

        :param im: Input image tensor
        :return: Converted image tensor
        """
        # There is not really a CIE D60 white point, but that's what everyone calls what ACES uses.
        return mm(im, cls.mat_d60_to_d65)

    @classmethod
    def _d65_to_d60(cls, im:torch.Tensor) -> torch.Tensor:
        """
        Convert CIE XYZ image from D65 to "D60" white point.

        :param torch.Tensor im: Input image tensor
        :return: Converted image tensor
        """
        return mm(im, cls.mat_d65_to_d60)

class TransferFunction:    
    """
    Opto-electronic/electro-optical transfer functions. Example:

    .. highlight:: python
    .. code-block:: python
        
        im_srgb = TransferFunction.srgb_oetf(im_linear)

    .. note::
        These transfer functions are applied automatically by :code:`ColorSpace.convert` when appropriate, 
        but can instead be used explicitly.

    """
    @staticmethod
    def srgb_eotf(im:torch.Tensor) -> torch.Tensor:
        """
        sRGB electro-optical transfer function (sRGB gamma to linear sRGB)

        :param im: sRGB image tensor 
        :return: linear sRGB image tensor
        """
        s1 = im / 12.92321
        s2 = torch.pow((im + 0.055) / 1.055, 12. / 5)
        return torch.where(im <= 0.04045, s1, s2)

    @staticmethod
    def srgb_oetf(im:torch.Tensor) -> torch.Tensor:
        """
        sRGB opto-electronic transfer function (linear sRGB to sRGB gamma)

        :param im: linear sRGB image tensor 
        :return: sRGB image tensor
        """
        s1 = im * 12.92321
        s2 = torch.pow(im, 1. / 2.4) * 1.055 - 0.055
        return torch.where(im <= 0.0031308, s1, s2)

    @staticmethod
    def rec709_eotf(im:torch.Tensor) -> torch.Tensor:
        """
        Rec. 709 electro-optical transfer function (Rec. 709 gamma to linear sRGB)

        :param im: Rec. 709 image tensor 
        :return: linear sRGB image tensor (same primaries)
        """
        s1 = im / 4.5
        s2 = torch.pow((im + 0.099) / 1.099, 2.2)
        return torch.where(im <= 0.081, s1, s2)

    @staticmethod
    def rec709_oetf(im:torch.Tensor) -> torch.Tensor:
        """
        Rec. 709 opto-electronic transfer function (linear sRGB to Rec. 709 gamma)

        :param im: linear sRGB image tensor (same primaries)
        :return: Rec. 709 image tensor
        """
        s1 = im * 4.5
        s2 = torch.pow(im, .4545) * 1.099 - 0.099
        return torch.where(im <= 0.018, s1, s2)

    @staticmethod
    def rec2020_eotf(im:torch.Tensor) -> torch.Tensor:
        """
        Rec. 2020 electro-optical transfer function (Rec. 2020 gamma to linear)

        :param im: Rec. 2020 image tensor 
        :return: linear Rec. 2020 gamut image tensor
        """
        a = 1.09929682680944
        b = 0.08124285829     
        s1 = im / 4.5
        s2 = torch.pow((im + a - 1.) / a, 1./ 0.45)
        return torch.where(im <= b, s1, s2)

    @staticmethod
    def rec2020_oetf(im:torch.Tensor) -> torch.Tensor: 
        """
        Rec. 2020 opto-electronic transfer function (linear to Rec. 2020 gamma)

        :param im: linear Rec. 2020 gamut image tensor 
        :return: Rec. 2020 image tensor
        """
        a = 1.09929682680944
        b = 0.018053968510807
        s1 = im * 4.5
        s2 = a * torch.pow(im, .45) - (a - 1.)
        return torch.where(im <= b, s1, s2)

    @staticmethod
    def dcip3_eotf(im:torch.Tensor) -> torch.Tensor: 
        """
        DCI P3 electro-optical transfer function (DCI P3 gamma to linear)

        :param im: DCI P3 image tensor 
        :return: linear P3 gamut image tensor
        """
        return torch.pow(im, 2.6)

    @staticmethod
    def dcip3_oetf(im:torch.Tensor) -> torch.Tensor: 
        """
        DCI P3 opto-electronic transfer function (linear to DCI P3 gamma)

        :param im: linear P3 gamut image tensor 
        :return: DCI P3 image tensor
        """
        return torch.pow(im, 1./2.6)

    @staticmethod
    def log_c_eotf(im:torch.Tensor) -> torch.Tensor:
        """
        LogC electro-optical transfer function

        :param im: LogC encoded image tensor
        :return: linear image tensor 
        """
        offset = 0.00937677
        x = im.clone()
        x = torch.where(x > 0.1496582, 
            torch.pow(10.0, (x - 0.385537) / 0.2471896) * 0.18 - offset,
            (x / 0.9661776 - 0.04378604) * 0.18 - offset)
        return x

    @staticmethod
    def log_c_oetf(im:torch.Tensor) -> torch.Tensor:
        """
        LogC opto-electronic transfer function

        :param im: linear image tensor 
        :return: LogC encoded image tensor
        """
        offset = 0.00937677
        x = im.clone()
        x = torch.where(x > 0.02 - offset,
            (((torch.log10((x + offset) / 0.18)) * 0.2471896) + 0.385537),
            ((((x + offset) / 0.18) + 0.04378604) * 0.9661776))
        return x

    @staticmethod
    def s_log_eotf(im:torch.Tensor) -> torch.Tensor:
        """
        S-Log electro-optical transfer function

        :param im: S-Log encoded image tensor
        :return: linear image tensor 
        """
        x = im.clone()
        return torch.pow(10.0, ((x - 0.616596 - 0.03) / 0.432699)) - 0.037584

    @staticmethod
    def s_log_oetf(im:torch.Tensor) -> torch.Tensor:
        """
        S-Log opto-electronic transfer function

        :param im: linear image tensor 
        :return: S-Log encoded image tensor
        """
        x = im.clone()
        return (0.432699 * torch.log10(x + 0.037584) + 0.616596) + 0.03