"""Image data definitions, both general and renderer-specific"""
import typing
from enum import IntEnum

class GraphicsFormat(IntEnum):
    """
    The graphics format of an image file to be saved or loaded. Available options are:
    
    .. highlight:: text
    .. code-block:: text

        - UNKNOWN
        - R8_UINT             
        - R8G8_UINT           
        - R8G8B8_UINT         
        - R8G8B8A8_UINT       
        - R16_UINT            
        - R16G16_UINT         
        - R16G16B16_UINT      
        - R16G16B16A16_UINT   
        - R32_UINT            
        - R32G32_UINT         
        - R32G32B32_UINT      
        - R32G32B32A32_UINT   
        - R16_SFLOAT          
        - R16G16_SFLOAT       
        - R16G16B16_SFLOAT    
        - R16G16B16A16_SFLOAT 
        - R32_SFLOAT          
        - R32G32_SFLOAT       
        - R32G32B32_SFLOAT    
        - R32G32B32A32_SFLOAT 
    """
    UNKNOWN             = 1<<0
    R8_UINT             = 1<<1
    R8G8_UINT           = 1<<2
    R8G8B8_UINT         = 1<<3
    R8G8B8A8_UINT       = 1<<4
    R16_UINT            = 1<<5
    R16G16_UINT         = 1<<6
    R16G16B16_UINT      = 1<<7
    R16G16B16A16_UINT   = 1<<8
    R32_UINT            = 1<<9
    R32G32_UINT         = 1<<10
    R32G32B32_UINT      = 1<<11
    R32G32B32A32_UINT   = 1<<12
    R16_SFLOAT          = 1<<13
    R16G16_SFLOAT       = 1<<14
    R16G16B16_SFLOAT    = 1<<15
    R16G16B16A16_SFLOAT = 1<<16
    R32_SFLOAT          = 1<<17
    R32G32_SFLOAT       = 1<<18
    R32G32B32_SFLOAT    = 1<<19
    R32G32B32A32_SFLOAT = 1<<20

    READABLE = R8_UINT | R8G8_UINT | R8G8B8_UINT | R8G8B8A8_UINT | \
        R16_UINT | R16G16_UINT | R16G16B16_UINT | R16G16B16A16_UINT | \
        R32_UINT | R32G32_UINT | R32G32B32_UINT | R32G32B32A32_UINT | \
        R16_SFLOAT | R16G16_SFLOAT | R16G16B16_SFLOAT | R16G16B16A16_SFLOAT | \
        R32_SFLOAT | R32G32_SFLOAT | R32G32B32_SFLOAT | R32G32B32A32_SFLOAT

    UINT8 = R8_UINT | R8G8_UINT | R8G8B8_UINT | R8G8B8A8_UINT
    UINT16 = R16_UINT | R16G16_UINT | R16G16B16_UINT | R16G16B16A16_UINT
    UINT32 = R32_UINT | R32G32_UINT | R32G32B32_UINT | R16G16B16A16_UINT
    SFLOAT16 = R16_SFLOAT | R16G16_SFLOAT | R16G16B16_SFLOAT | R16G16B16A16_SFLOAT
    SFLOAT32 = R32_SFLOAT | R32G32_SFLOAT | R32G32B32_SFLOAT | R32G32B32A32_SFLOAT

    WRITABLE_PNG = UINT8 | UINT16
    WRITABLE_TIF = SFLOAT16 | SFLOAT32
    WRITABLE_EXR = SFLOAT16 | SFLOAT32

    C1 = R8_UINT | R16_UINT | R32_UINT | R16_SFLOAT | R32_SFLOAT
    C2 = R8G8_UINT | R16G16_UINT | R32G32_UINT | R16G16_SFLOAT | R32G32_SFLOAT
    C3 = R8G8B8_UINT | R16G16B16_UINT | R32G32B32_UINT | R16G16B16_SFLOAT | R32G32B32_SFLOAT
    C4 = R8G8B8A8_UINT | R16G16B16A16_UINT | R32G32B32A32_UINT | R16G16B16A16_SFLOAT | R32G32B32A32_SFLOAT

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