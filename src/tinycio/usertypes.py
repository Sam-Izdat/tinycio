from __future__ import annotations
import typing
from typing import Union
import os

import torch
import numpy as np

from .numerics import Float2, Float3
from .colorspace import ColorSpace
from .balance import WhiteBalance
from .tonemapping import ToneMapping
from .lut import LookupTable
from .fsio.imagefile import _infer_image_file_format, load_image, save_image, truncate_image
from .fsio.format import GraphicsFormat, ImageFileFormat, LUTFormat
from .correction import ColorCorrection
from .util.colorutil import apply_hue_oklab, apply_gamma, col_okhsv_to_srgb, col_srgb_to_okhsv, \
    col_okhsl_to_srgb, col_srgb_to_okhsl

# NOTE: This module SHOULD NOT be imported by other internal modules. 
# It is a user-facing abstraction layer. The rest of the code should be oblivious to it. 
# Typing can use __future__.annotation so that its types can be listed in the docs where applicable.

class Chromaticity(Float2):
    """
    2-component float32 CIE xy chromaticity type. Convenience wrapper around :class:`.Float2`, 
    which is in turn a shape [2] numpy.ndarray.


    :param option1_param0: x component
    :type option1_param0: float
    :param option1_param1: y component
    :type option1_param1: float

    :param option2_param0: value of both components
    :type option2_param0: float

    :param option3_param0: 2-component tensor or array 
        (a 2D tensor/array image is allowed if the number of pixel values is 1)
    :type option3_param0: torch.Tensor | numpy.ndarray | Float2
    """
    def __new__(cls, *args, **kwargs):
        return super(Chromaticity, cls).__new__(cls, *args, **kwargs)

    def to_xyy(self, luminance:float=0.) -> Color:
        """
        Returns CIE xyY color

        :param luminance: Y luminance
        :return: CIE xyY color
        """
        return Color(self[0], self[1], luminance)

    def to_xyz(self, luminance:float=0.) -> Color:
        """
        Returns CIE XYZ color

        :param luminance: Y luminance
        :return: CIE XYZ color
        """

        x = (luminance / self[1]) * self[0]
        z = (luminance / self[1]) * (1 - self[0] - self[1])
        return Color(x, luminance, z)

def wrap_pt_im_method(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        return self.__class__(result, color_space=self.color_space) if isinstance(result, torch.Tensor) else result
    return wrapper

class ColorImage(torch.Tensor):
    """
    2D 3-channel color image using float32. Convenience wrapper around a [C=3, H, W] sized PyTorch image tensor. 
    This is the most hands-off abstraction, as it automatically handles color space conversion as needed.

    .. note::
        All transformative methods will create a new :class:`ColorImage` instance without altering the 
        current state. PyTorch methods and operations will work as expected but will return with 
        the color space set to UNKNOWN. Exceptions include :code:`.clamp()`, :code:`.lerp()`, 
        :code:`.minimum()`, and :code:`.maximum()`, which will retain the current state for convenience.

    .. note:: 
        Assumes size/shape [C, H, W] for both PyTorch tensor or NumPy array inputs.

    :param param0: [C=3, H, W] sized tensor/array
    :type param0: torch.Tensor | numpy.ndarray
    :param color_space: the color space of the image
    :type color_space: str | ColorSpace.Variant

    """
    color_space = ColorSpace.Variant.UNKNOWN
    
    meta_channel_names  = ['unknown', 'unknown', 'uknown']
    meta_white_point    = 'unknown'
    meta_gamut          = 'unknown'
    meta_peceptual      = False
    meta_negative       = True
    meta_scene_linear   = False

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], np.ndarray) and len(args[0].shape) == 3 and args[0].shape[0] == 3:
            ten = torch.from_numpy(args[0]).float()
        elif torch.is_tensor(args[0]) and len(args[0].size()) == 3 and args[0].size(0) == 3:
            ten = args[0].float().clone().detach().requires_grad_(False)
        else:
            raise TypeError("ColorImage must be a [C=3, H, W] sized image tensor/array.")

        # There is a reason why this has to be here, in this order, and why this hangs "color_space" on "new".
        # Setting it prematurely on cls will yeet the metadata of the instance being transitioned to a new cs.
        new = super(ColorImage, cls).__new__(cls, ten)
        cs_default = args[0].color_space if isinstance(args[0], ColorImage) else ColorSpace.Variant.UNKNOWN
        new.color_space = args[1] if len(args) > 1 else kwargs.get("color_space", cs_default)
        if type(new.color_space) is str: new.color_space = ColorSpace.Variant[new.color_space.strip().upper()]
        assert isinstance(new.color_space, ColorSpace.Variant), "Invalid color space"
        new._set_meta()
        return new

    # We want to maintain state through these common torch methods.
    # Annoying, but __torch_function__ is footgun city, and I'm not sure it would work anyway.
    @wrap_pt_im_method
    def clamp(self, *args, **kwargs):
        return super().clamp(*args, **kwargs)

    @wrap_pt_im_method
    def clip(self, *args, **kwargs):
        return super().clamp(*args, **kwargs)

    @wrap_pt_im_method
    def lerp(self, *args, **kwargs):
        return super().lerp(*args, **kwargs)
        
    @wrap_pt_im_method
    def minimum(self, *args, **kwargs):
        return super().minimum(*args, **kwargs)
        
    @wrap_pt_im_method
    def maximum(self, *args, **kwargs):
        return super().maximum(*args, **kwargs)

    def _set_meta(self):
        # Meta variables are only used for a debug printout at this time
        if not self.color_space: return
        cs = ColorSpace.Variant

        if self.color_space & cs.MODEL_RGB:
            self.meta_channel_names  = ['red', 'green', 'blue']
        elif self.color_space == cs.CIE_XYZ:
            self.meta_channel_names  = ['X', 'Y', 'Z']
        elif self.color_space == cs.CIE_XYY:
            self.meta_channel_names  = ['x', 'y', 'Y']
        elif self.color_space == cs.LMS:
            self.meta_channel_names  = ['long', 'medium', 'short']
        elif self.color_space & (cs.OKLAB | cs.CIELAB):
            self.meta_channel_names  = ['L*', 'a*', 'b*']
        elif self.color_space == cs.CIELUV:
            self.meta_channel_names  = ['L*', 'u*', 'v*']
        elif self.color_space & (cs.HSL | cs.OKHSL):
            self.meta_channel_names  = ['hue', 'saturation', 'lightness']
        elif self.color_space & (cs.HSV | cs.OKHSV):
            self.meta_channel_names  = ['hue', 'saturation', 'value']
        else:
            self.meta_channel_names  = ['unknown', 'unknown', 'unknown']

        if self.color_space & cs.GAMUT_SRGB:
            self.meta_gamut = "sRGB"
        elif self.color_space & cs.GAMUT_AP0:
            self.meta_gamut = "AP0"
        elif self.color_space & cs.GAMUT_AP1:
            self.meta_gamut = "AP1"
        elif self.color_space & cs.GAMUT_REC2020:
            self.meta_gamut = "Rec. 2020 / BT. 2020"
        elif self.color_space & cs.GAMUT_DCI_P3:
            self.meta_gamut = "DCI-P3"
        elif self.color_space & cs.GAMUT_DISPLAY_P3:
            self.meta_gamut = "Display P3"
        elif self.color_space & cs.GAMUT_OKLAB:
            self.meta_gamut = "OKLAB"
        elif self.color_space & cs.GAMUT_CIE_XYZ:
            self.meta_gamut = "CIE 1931 XYZ"
        elif self.color_space & cs.GAMUT_CIELAB:
            self.meta_gamut = "CIELAB"
        elif self.color_space & cs.GAMUT_CIELUV:
            self.meta_gamut = "CIELUV"
        elif self.color_space & cs.GAMUT_OTHER:
            self.meta_gamut = "other"
        else:
            self.meta_gamut = "unknown"

        if self.color_space & cs.WP_D65:
            self.meta_white_point    = 'D65'
        elif self.color_space & cs.WP_CCT_6300:
            self.meta_white_point    = '~6300K'
        elif self.color_space & cs.WP_CCT_6000:
            self.meta_white_point    = '~6000K (a.k.a. "D60")'
        else:
            self.meta_white_point    = "other/unknown/non-applicable"

        self.meta_scene_linear  = self.color_space & cs.SCENE_LINEAR
        self.meta_peceptual     = self.color_space & cs.PERCEPTUAL
        self.meta_negative      = self.color_space & cs.NEGATIVE


    def info(self, printout=True) -> Union[bool, str]:
        """
        Print or return a string describing the properties of the image for debugging.

        :param printout: Print string if True, return string if False
        """
        infostr = "IMAGE DESCRIPTION"
        infostr += "\n================="
        infostr += "\nCOLOR SPACE          " + str(ColorSpace.Variant(self.color_space).name)
        infostr += "\nGAMUT                " + str(self.meta_gamut)
        infostr += "\nWHITE POINT          " + str(self.meta_white_point)
        infostr += "\nSCENE-LINEAR         " + ("YES" if self.meta_scene_linear else "NO/unknown/non-applicable")
        infostr += "\nPERCEPTUAL           " + ("YES" if self.meta_peceptual else "NO")
        infostr += "\nCAN GO NEGATIVE      " + ("YES" if self.meta_negative else "NO")
        infostr += "\nCHANNELS             " + str(self.size(0))
        infostr += "\nCHANNEL NAMES        " + str(", ".join(self.meta_channel_names))
        infostr += "\nWIDTH                " + str(self.size(2))
        infostr += "\nHEIGHT               " + str(self.size(1))
        infostr += "\nDTYPE                " + str(self.dtype)
        if printout: 
            print(infostr)
            return True
        else:
            return infostr

    @staticmethod
    def load(
        fp:str, 
        color_space:Union[str, ColorSpace.Variant]=ColorSpace.Variant.UNKNOWN,
        graphics_format:Union[str, GraphicsFormat]=GraphicsFormat.UNKNOWN) -> ColorImage:
        """
        Load image from file

        :param fp: image file path
        :param color_space: image color space
        :type color_space: str | ColorSpace.Variant
        :param graphics_format: image graphics format
        :type graphics_format: str | fsio.GraphicsFormat
        :return: New :class:`ColorImage` loaded from file.
        """
        fp = os.path.realpath(fp)
        fn, fnext = os.path.splitext(fp)
        ext = _infer_image_file_format(fnext)

        if type(graphics_format) is str: graphics_format = GraphicsFormat[graphics_format.strip().upper()]
        assert isinstance(graphics_format, GraphicsFormat), "Invalid graphics format"

        if type(color_space) is str: color_space = ColorSpace.Variant[color_space.strip().upper()]
        assert isinstance(color_space, ColorSpace.Variant), "Invalid color space"

        if graphics_format == GraphicsFormat.UNKNOWN:
            if ext & ImageFileFormat.UINT8:
                # if file format is conventionally 8 bpc, assume uint8
                graphics_format = GraphicsFormat.UINT8
            else:
                # otherwise assume float32
                graphics_format = GraphicsFormat.SFLOAT32
        if color_space == ColorSpace.Variant.UNKNOWN:
            if ext & ImageFileFormat.UINT8:
                # if file format is conventionally 8 bpc, assume sRGB gamma curve
                color_space=ColorSpace.Variant.SRGB
            else:
                # otherwise assume sRGB linear
                color_space=ColorSpace.Variant.SRGB_LIN

        im = load_image(fp, graphics_format=graphics_format)
        return ColorImage(truncate_image(im), color_space=color_space)

    def save(self, fp:str, graphics_format:Union[str, GraphicsFormat]=GraphicsFormat.UNKNOWN) -> bool:
        """
        Save image to file
        
        .. warning:: 
            This will overwrite existing files.

        :param fp: image file path
        :param graphics_format: image graphics format
        :type graphics_format: str | fsio.GraphicsFormat
        :return: True if successful
        """
        fp = os.path.realpath(fp)
        dp = os.path.dirname(fp)
        fn, fnext = os.path.splitext(fp)
        ext = _infer_image_file_format(fnext)

        if not os.path.isdir(dp): raise Exception(f"directory {dp} does not exist")

        if type(graphics_format) is str: graphics_format = GraphicsFormat[graphics_format.strip().upper()]
        assert isinstance(graphics_format, GraphicsFormat), "Invalid graphics format"

        save_image(self.clone(), fp, graphics_format=graphics_format)
        return True

    def to_color_space(self, color_space:Union[str, ColorSpace.Variant]) -> ColorImage:
        """
        Convert the image to a different color space and return as new :class:`ColorImage`.

        :param color_space: the destination color space
        :type color_space: str | ColorSpace.Variant
        :return: New :class:`ColorImage` in designated color space.
        """
        if type(color_space) is str: color_space = ColorSpace.Variant[color_space.strip().upper()]
        assert isinstance(color_space, ColorSpace.Variant), "Invalid color space"
        res = None
        res = ColorSpace.convert(self.clone(), source=self.color_space, destination=color_space)
        res = ColorImage(res, color_space=color_space)
        return res

    def tone_map(self, tone_mapper:Union[str,ToneMapping.Variant]) -> ColorImage:
        """
        Apply tone mapping and return as new :class:`ColorImage`.

        .. note::
            Any needed color space conversion will be handled automatically.

        :param tone_mapper: the tone mapper to use
        :type tone_mapper: str | ToneMapping.Variant
        :return: New tone mapped :class:`ColorImage`.
        """
        if type(tone_mapper) is str: tone_mapper = ToneMapping.Variant[tone_mapper.strip().upper()]
        assert isinstance(tone_mapper, ToneMapping.Variant), "Invalid tone mapper"
        ip = self.clone()
        cs_tm = self.color_space
        if tone_mapper == ToneMapping.Variant.ACESCG:
            cs_tm = ColorSpace.Variant.ACESCG
        else:
            cs_tm = ColorSpace.Variant.SRGB_LIN
        res = ColorSpace.convert(ip, source=self.color_space, destination=cs_tm)
        res = ToneMapping.apply(res, tone_mapper=tone_mapper)
        res = ColorSpace.convert(res, source=cs_tm, destination=self.color_space)
        return ColorImage(res, color_space=self.color_space)

    def lut(self, lut:Union[str, LookupTable], lut_format=LUTFormat.UNKNOWN) -> ColorImage:
        """
        Apply a :class:`LookupTable` and return as new :class:`ColorImage`.

        :param lut: :class:`.LookupTable` or LUT file path
        :param lut_format: Format of the LUT, if loading from file.
        :return: New :class:`ColorImage` with LUT applied.
        """
        if isinstance(lut, LookupTable):
            res = lut.apply(self.clone())
        elif type(lut) == str:
            lut = LookupTable.load(lut, lut_format=lut_format)
            res = lut.apply(self.clone())

        return ColorImage(res, color_space=self.color_space)

    def white_balance(self, 
        source_white:Union[str, Float2, Chromaticity, float, WhiteBalance.Illuminant]="auto", 
        target_white:Union[str, Float2, Chromaticity, float, WhiteBalance.Illuminant]=WhiteBalance.Illuminant.NONE) -> ColorImage:
        """
        White balance the image and return as new :class:`ColorImage`.

        .. note::
            Any needed color space conversion will be handled automatically.
            
        :param source_white: Source white point - can be any of: 

            * *string* "auto" for automatic (:py:meth:`~WhiteBalance.wp_from_image`)
            * *int* for correlated color temperature (:py:meth:`~WhiteBalance.wp_from_cct`)
            * :class:`~WhiteBalance.Illuminant` or matching *string* (:py:meth:`~WhiteBalance.wp_from_illuminant`)
            * CIE 1931 xy coordinates as :class:`.Float2`/:class:`.Chromaticity`

        :param target_white: Target white point - can be any of:

            * *int* for correlated color temperature (:py:meth:`~WhiteBalance.wp_from_cct`)
            * :class:`~WhiteBalance.Illuminant` or matching *string* (:py:meth:`~WhiteBalance.wp_from_illuminant`)
            * CIE 1931 xy coordinates as :class:`.Float2`/:class:`.Chromaticity`

        :return: New :class:`ColorImage` with white balance applied.
        """
        assert target_white is not WhiteBalance.Illuminant.NONE, "target white point cannot be NONE"
        if type(source_white) == str and source_white.strip().upper() == "AUTO":
            source_white = WhiteBalance.wp_from_image(self.to_color_space(ColorSpace.Variant.CIE_XYZ))
        elif type(source_white) in [int, float]:
            source_white = WhiteBalance.wp_from_cct(source_white)
        elif isinstance(source_white, WhiteBalance.Illuminant):
            source_white = WhiteBalance.wp_from_illuminant(source_white)
        elif type(source_white) == str:
            source_white = WhiteBalance.Illuminant[source_white]
            source_white = WhiteBalance.wp_from_illuminant(source_white)
        elif isinstance(source_white, Float2):
            pass
        else:
            raise Exception("could not interpret source white point")

        if type(target_white) in [int, float]:
            target_white = WhiteBalance.wp_from_cct(target_white)
        elif isinstance(target_white, WhiteBalance.Illuminant):
            target_white = WhiteBalance.wp_from_illuminant(target_white)
        elif type(target_white) == str:
            target_white = WhiteBalance.Illuminant[target_white]
            target_white = WhiteBalance.wp_from_illuminant(target_white)
        elif isinstance(target_white, Float2):
            pass
        else:
            raise Exception("could not interpret target white point")
        assert isinstance(source_white, Float2) and isinstance(target_white, Float2), "failed to acquire valid white point"

        im_lms = ColorSpace.convert(self.clone(), source=self.color_space, destination=ColorSpace.Variant.LMS)
        im_lms = WhiteBalance.apply(im_lms, source_white=source_white, target_white=target_white)
        res = ColorSpace.convert(im_lms, source=ColorSpace.Variant.LMS, destination=self.color_space)
        return ColorImage(res, color_space=self.color_space)

    def correct(self, cc:ColorCorrection) -> ColorImage:
        """
        Apply color correction and return as new :class:`ColorImage`.

        :param cc: Color correction object.
        :return: New :class:`ColorImage` with color correction applied.
        """
        cs_acescc = ColorSpace.Variant.ACESCC
        im = ColorSpace.convert(self.clone(), source=self.color_space, destination=cs_acescc)
        res = cc.apply(im)
        res = ColorSpace.convert(res, source=cs_acescc, destination=self.color_space)
        return ColorImage(res, color_space=self.color_space)

class MonoImage(torch.Tensor):
    """
    2D monochromatic image using float32. Convenience wrapper around a [C=1, H, W] sized PyTorch image tensor. 
    This is a utility class that has no built-in functionality beyond enforcing a single channel for initial inputs. 
    Can be converted to :class:`ColorImage` with :code:`ColorImage(mono_im.repeat(3,1,1))`.

    .. note:: 
        Assumes size/shape [C, H, W] for both PyTorch tensor or NumPy array inputs.

    :param param0: [C=1, H, W] sized tensor/array
    :type param0: torch.Tensor | numpy.ndarray

    """
    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], np.ndarray) and len(args[0].shape) == 3 and args[0].shape[0] == 1:
            ten = torch.from_numpy(args[0]).float()
        elif torch.is_tensor(args[0]) and len(args[0].size()) == 3 and args[0].size(0) == 1:
            ten = args[0].float().clone().detach().requires_grad_(False)
        else:
            raise TypeError("MonoImage must be a [C=1, H, W] sized image tensor/array.")
        return super(MonoImage, cls).__new__(cls, ten)

    def info(self, printout=True) -> Union[bool, str]:
        """
        Print or return a string describing the properties of the image for debugging.

        :param printout: Print string if True, return string if False
        """
        infostr = "IMAGE DESCRIPTION"
        infostr += "\n================="
        infostr += "\nCLASS              " + str(self.__class__.__name__)
        infostr += "\nCOLOR SPACE          N/A"
        infostr += "\nCHANNELS             " + str(self.size(0))
        infostr += "\nWIDTH                " + str(self.size(2))
        infostr += "\nHEIGHT               " + str(self.size(1))
        infostr += "\nDTYPE                " + str(self.dtype)
        if printout: 
            print(infostr)
            return True
        else:
            return infostr

class Color(Float3):
    """
    3-component float32 color type. Convenience wrapper around :class:`.Float3`, 
    which is in turn a shape [3] numpy.ndarray. Example:

    .. highlight:: python
    .. code-block:: python

        a = Color(1, 2, 3)
        b = Color(torch.tensor([1, 2, 3]))
        c = Color(numpy.array([1, 2, 3]))
        d = Color(torch.tensor([1, 2, 3]).unsqueeze(-1).unsqueeze(-1))
        numpy.array_equal(a,b) & numpy.array_equal(b,c) & numpy.array_equal(c,d) # True

    .. note:: 
        This data type is color space agnostic until a :class:`.ColorImage` is requested. 
        Swizzling is allowed, but a numeric type will be returned if the number of 
        swizzled components is not 3.

    :param option1_param0: x/r component
    :type option1_param0: float
    :param option1_param1: y/g component
    :type option1_param1: float
    :param option1_param2: z/b component
    :type option1_param2: float

    :param option2_param0: value of every component
    :type option2_param0: float

    :param option3_param0: 3-component tensor or array 
        (a 2D tensor/array image is allowed if the number of pixel values is 1)
    :type option3_param0: torch.Tensor | numpy.ndarray | Float3

    """
    def __new__(cls, *args, **kwargs):
        return super(Color, cls).__new__(cls, *args, **kwargs)

    def convert(self, 
        source:Union[str, ColorSpace.Variant], 
        destination:Union[str, ColorSpace.Variant]) -> Color:
        """
        Convert color value from one color space to another.

        :param source: color space to convert from
        :type source: str | ColorSpace.Variant
        :param destination: color space to convert to
        :type destination: str | ColorSpace.Variant
        """
        if source == destination: return self

        if type(source) is str: source = ColorSpace.Variant[source.strip().upper()]
        assert isinstance(source, ColorSpace.Variant), "Invalid source color space"
        if type(destination) is str: destination = ColorSpace.Variant[destination.strip().upper()]
        assert isinstance(destination, ColorSpace.Variant), "Invalid source color space"

        col = self.copy()
        # This is dirty, but we need to check for what's unsupported by ColorSpace.convert
        if (source & ColorSpace.Variant.SUPPORTED) and (destination & ColorSpace.Variant.SUPPORTED):
            # Both source and destination supported by ColorSpace.convert - the general case
            im_col = self.image(color_space=source)
            im_col = im_col.to_color_space(destination)
            return Color(im_col)
        elif source == ColorSpace.Variant.SRGB_LIN and destination == ColorSpace.Variant.OKHSV:
            return Color(col_srgb_to_okhsv(col))
        elif source == ColorSpace.Variant.SRGB_LIN and destination == ColorSpace.Variant.OKHSL:
            return Color(col_srgb_to_okhsl(col))
        elif source == ColorSpace.Variant.OKHSV and destination == ColorSpace.Variant.SRGB_LIN:
            return Color(col_okhsv_to_srgb(col))
        elif source == ColorSpace.Variant.OKHSL and destination == ColorSpace.Variant.SRGB_LIN:
            return Color(col_okhsl_to_srgb(col))
        elif source == ColorSpace.Variant.OKHSV:
            # Reroute to SRGB_LIN-to-X
            col = Color(col_okhsv_to_srgb(col))
            return col.convert(source=ColorSpace.Variant.SRGB_LIN, destination=destination)
        elif source == ColorSpace.Variant.OKHSL:
            # Reroute to SRGB_LIN-to-X
            col = Color(col_okhsl_to_srgb(col))
            return col.convert(source= ColorSpace.Variant.SRGB_LIN, destination=destination)
        elif (source & ColorSpace.Variant.SUPPORTED) and (destination & ColorSpace.Variant.UNSUPPORTED):
            # Reroute to recursive X-to-SRGB
            col = col.convert(source=source, destination=ColorSpace.Variant.SRGB_LIN)
            return col.convert(source=ColorSpace.Variant.SRGB_LIN, destination=destination)
        else:
            raise Exception("unsupported color conversion")
        return False


    def image(self, color_space:Union[str, ColorSpace.Variant]=ColorSpace.Variant.UNKNOWN) -> ColorImage:
        """
        Unsqueeze to a [C=3, H, W] sized PyTorch image tensor.

        :param color_space: the color space of the image
        :type color_space: str | ColorSpace.Variant
        :return: Expanded [C=3, H=1, W=1] float32 "image" tensor
        """
        if type(color_space) is str: color_space = ColorSpace.Variant[color_space.strip().upper()]
        assert isinstance(color_space, ColorSpace.Variant), "Invalid color space"
        ten = torch.tensor([self[0], self[1], self[2]], dtype=torch.float32)
        return ColorImage(ten.unsqueeze(-1).unsqueeze(-1), color_space=color_space)

    # We intercept these:
    @property
    def xxx(self): return Color(self.x, self.x, self.x)
    @property
    def xxy(self): return Color(self.x, self.x, self.y)
    @property
    def xxz(self): return Color(self.x, self.x, self.z)
    @property
    def xyx(self): return Color(self.x, self.y, self.x)
    @property
    def xyy(self): return Color(self.x, self.y, self.y)
    @property
    def xyz(self): return self
    @property
    def xzx(self): return Color(self.x, self.z, self.x)
    @property
    def xzy(self): return Color(self.x, self.z, self.y)
    @property
    def xzz(self): return Color(self.x, self.z, self.z)
    @property
    def yxx(self): return Color(self.y, self.x, self.x)
    @property
    def yxy(self): return Color(self.y, self.x, self.y)
    @property
    def yxz(self): return Color(self.y, self.x, self.z)
    @property
    def yyx(self): return Color(self.y, self.y, self.x)
    @property
    def yyy(self): return Color(self.y, self.y, self.y)
    @property
    def yyz(self): return Color(self.y, self.y, self.z)
    @property
    def yzx(self): return Color(self.y, self.z, self.x)
    @property
    def yzy(self): return Color(self.y, self.z, self.y)
    @property
    def yzz(self): return Color(self.y, self.z, self.z)
    @property
    def zxx(self): return Color(self.z, self.x, self.x)
    @property
    def zxy(self): return Color(self.z, self.x, self.y)
    @property
    def zxz(self): return Color(self.z, self.x, self.z)
    @property
    def zyx(self): return Color(self.z, self.y, self.x)
    @property
    def zyy(self): return Color(self.z, self.y, self.y)
    @property
    def zyz(self): return Color(self.z, self.y, self.z)
    @property
    def zzx(self): return Color(self.z, self.z, self.x)
    @property
    def zzy(self): return Color(self.z, self.z, self.y)
    @property
    def zzz(self): return Color(self.z, self.z, self.z)

    @property
    def rrr(self): return Color(self.r, self.r, self.r)
    @property
    def rrg(self): return Color(self.r, self.r, self.g)
    @property
    def rrb(self): return Color(self.r, self.r, self.b)
    @property
    def rgr(self): return Color(self.r, self.g, self.r)
    @property
    def rgg(self): return Color(self.r, self.g, self.g)
    @property
    def rgb(self): return self
    @property
    def rbr(self): return Color(self.r, self.b, self.r)
    @property
    def rbg(self): return Color(self.r, self.b, self.g)
    @property
    def rbb(self): return Color(self.r, self.b, self.b)
    @property
    def grr(self): return Color(self.g, self.r, self.r)
    @property
    def grg(self): return Color(self.g, self.r, self.g)
    @property
    def grb(self): return Color(self.g, self.r, self.b)
    @property
    def ggr(self): return Color(self.g, self.g, self.r)
    @property
    def ggg(self): return Color(self.g, self.g, self.g)
    @property
    def ggb(self): return Color(self.g, self.g, self.b)
    @property
    def gbr(self): return Color(self.g, self.b, self.r)
    @property
    def gbg(self): return Color(self.g, self.b, self.g)
    @property
    def gbb(self): return Color(self.g, self.b, self.b)
    @property
    def brr(self): return Color(self.b, self.r, self.r)
    @property
    def brg(self): return Color(self.b, self.r, self.g)
    @property
    def brb(self): return Color(self.b, self.r, self.b)
    @property
    def bgr(self): return Color(self.b, self.g, self.r)
    @property
    def bgg(self): return Color(self.b, self.g, self.g)
    @property
    def bgb(self): return Color(self.b, self.g, self.b)
    @property
    def bbr(self): return Color(self.b, self.b, self.r)
    @property
    def bbg(self): return Color(self.b, self.b, self.g)
    @property
    def bbb(self): return Color(self.b, self.b, self.b)