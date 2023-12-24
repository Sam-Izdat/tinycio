"""Image data definitions, both general and renderer-specific"""
import typing
from enum import IntEnum

class GraphicsFormat(IntEnum):
    """
    The graphics format of an image file to be saved or loaded. For a list of available options, see :ref:`ref_graphics_formats`.
    """
    UNKNOWN             = 1<<0
    UINT8               = 1<<1
    UINT16              = 1<<2
    UINT32              = 1<<3
    SFLOAT16            = 1<<4
    SFLOAT32            = 1<<5
    UNORM8              = 1<<6
    UNORM16             = 1<<7
    UNORM32             = 1<<8

    # Lump together any integer-type values
    I8 = UINT8 | UNORM8
    I16 = UINT16 | UNORM16
    I32 = UINT32 | UNORM32

    UNORM = UNORM8 | UNORM16 | UNORM32

    READABLE = UINT8 | UINT16 | UINT32 | UNORM8 | UNORM16 | UNORM32 | SFLOAT16 | SFLOAT32

    WRITABLE_PNG = UINT8 | UINT16 | UNORM8 | UNORM16
    WRITABLE_TIF = SFLOAT16 | SFLOAT32
    WRITABLE_EXR = SFLOAT16 | SFLOAT32

class ImageFileFormat(IntEnum):
    # TODO: Needs to be expanded after investigating iio support
    # NOTE: Not in user API right now, as it doesn't need to be
    UNKNOWN             = 1<<0
    PNG                 = 1<<1
    JPG                 = 1<<2
    EXR                 = 1<<3
    TIFF                = 1<<4
    WEBP                = 1<<5

    # This is annoying, so let's just...
    TIF = TIFF
    JPEG = JPG

    # Supported bit depth
    UINT8 = PNG | JPG | WEBP
    UINT16 = PNG
    SFLOAT16 = EXR | TIFF
    SFLOAT32 = EXR | TIFF

class LUTFormat(IntEnum):
    """
    Lookup table format. Available options are:

    .. highlight:: text
    .. code-block:: text
    
        - UNKNOWN
        - CUBE_3D
    """
    UNKNOWN     = 1<<0  # no color space specified - flag for guessing
    CUBE_3D     = 1<<1  # 3D CUBE LUT https://resolve.cafe/developers/luts/

    LUT_3D      = CUBE_3D # | etc for later