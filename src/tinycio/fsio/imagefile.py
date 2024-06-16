import torch
import numpy as np
import typing

import os
import imageio.v3 as iio

from .format import GraphicsFormat, ImageFileFormat

# import OpenEXR as exr
# from pathlib import Path
# import Imath
# import numpy as np

# def exr_to_array(filepath: Path):
#     exrfile = exr.InputFile(filepath)
#     print(len(exrfile.header()['channels']))
#     exit()
#     height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
#     width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
#     result = np.ndarray([3, height, width])
#     for i, chan in enumerate(['R', 'G', 'B']):
#         print(chan)
#         raw_bytes = exrfile.channel(chan, Imath.PixelType(Imath.PixelType.FLOAT))
#         depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
#         depth_map = np.reshape(depth_vector, (height, width))
#         result[i, :, :] = depth_map
#     return result


# def array_to_exr(arr):
#     width = arr.shape[2]
#     height = arr.shape[1]
#     size = width * height

#     FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

#     h = exr.Header(width, height)
#     h['channels'] = {'R' : Imath.Channel(FLOAT),
#         'G' : Imath.Channel(FLOAT), 
#         'B' : Imath.Channel(FLOAT)}
#     o = exr.OutputFile('./testy.exr', h)
#     r = arr[0, ...].astype(np.float32).tobytes('C')
#     g = arr[1, ...].astype(np.float32).tobytes('C') 
#     b = arr[2, ...].astype(np.float32).tobytes('C') 
#     # a = arr[2, ...].astype(np.float32).tobytes('C') 
#     channels = {'R' : r, 'G' : g, 'B' : b}
#     o.writePixels(channels)
#     o.close()

def _infer_image_file_format(ext:str) -> ImageFileFormat:
    ext = ext.strip().lower()
    if   ext == '.png':     return ImageFileFormat.PNG
    elif ext == '.jpg':     return ImageFileFormat.JPG
    elif ext == '.jpeg':    return ImageFileFormat.JPG
    elif ext == '.exr':     return ImageFileFormat.EXR
    elif ext == '.tif':     return ImageFileFormat.TIFF
    elif ext == '.tiff':    return ImageFileFormat.TIFF
    elif ext == '.webp':    return ImageFileFormat.WEBP
    else: return ImageFileFormat.UNKNOWN

def load_image(fp:str, graphics_format:GraphicsFormat=GraphicsFormat.UNKNOWN) -> torch.Tensor:
    """
    Load image file as [C, H, W] float32 PyTorch image tensor. 
    If the graphics format is not specified, we take a guess. Unsigned int  
    data is automatically converted to float32 in [0, 1] range. Floating 
    point data is converted to float32 if needed, but is otherwise undisturbed.

    :param str fp: File path to load from (including name and extension).
    :param GraphicsFormat graphics_format: Graphics format of the image.
    :return: [C, H, W] sized image tensor
    :rtype: torch.Tensor
    """
    fp = os.path.realpath(fp)
    fn, fnext = os.path.splitext(fp)
    out = None
    if graphics_format is None: graphics_format = GraphicsFormat.UNKNOWN

    # If not specified by user, assume UINT8 if png or jpg.  Otherwise, assume SFLOAT32. 
    file_fmt = _infer_image_file_format(fnext)
    denom = 1.
    if graphics_format & GraphicsFormat.I8:     denom = 255.
    elif graphics_format & GraphicsFormat.I16:  denom = 65535.
    elif graphics_format & GraphicsFormat.I32:  denom = 4294967295.
    elif graphics_format == GraphicsFormat.UNKNOWN and (file_fmt & ImageFileFormat.UINT8): denom = 255. # assume UINT8
    else: denom = 1.

    C = 0
    if file_fmt == ImageFileFormat.PNG:
        out = torch.from_numpy(iio.imread(fp, plugin="PNG-FI").astype(np.float32))
        if out.dim() == 2: out = out.unsqueeze(-1)
        out = out.permute(2,1,0).float()/denom
        C = out.size(0)
    else: 
        out = torch.from_numpy(iio.imread(fp).astype(np.float32))
        if out.dim() == 2: out = out.unsqueeze(-1)
        out = out.permute(2,1,0).float()/denom
        C = out.size(0)

    # Denormalize UNORM
    if graphics_format & GraphicsFormat.UNORM: 
        out = out.clamp(0., 1.) * 2. - 1.

    out = out.permute(0,2,1)
    if C < 1 or C > 4: raise Exception(f"Loaded image file with {C} channels; expected 1 to 4")

    return out

def save_image(im:torch.Tensor, fp:str, graphics_format:GraphicsFormat=GraphicsFormat.UNKNOWN) -> bool:
    """
    Save [C, H, W] float32 PyTorch image tensor as image file. 
    File format is inferred from extension by imageio. If a graphics format is not 
    specified, we take a guess.

    .. warning:: 
    
        This will overwrite existing files.

    :param torch.Tensor im: Image tensor.
    :param str fp: File path for saving the image (including name and extension).
    :param GraphicsFormat graphics_format: Graphics format of the image.
    :return: True if successful
    """
    fp = os.path.realpath(fp)
    fnwe = os.path.splitext(os.path.basename(fp))[0]
    fnext = os.path.splitext(fp)[-1]
    file_fmt = _infer_image_file_format(fnext)
    fac = 1.
    dtype = np.uint8
    if graphics_format & GraphicsFormat.I8:     
        fac = 255.
        dtype = np.uint8
    elif graphics_format & GraphicsFormat.I16:  
        fac = 65535.
        dtype = np.uint16
    elif graphics_format & GraphicsFormat.I32:  
        fac = 4294967295.
        dtype = np.uint32
    elif graphics_format & GraphicsFormat.SFLOAT16: 
        fac = 1.
        dtype = np.float16
    elif graphics_format & GraphicsFormat.SFLOAT32:
        fac = 1.
        dtype = np.float32
    elif graphics_format == GraphicsFormat.UNKNOWN and (file_fmt & ImageFileFormat.UINT8): 
        fac = 255.
        dtype = np.uint8
    else: 
        fac = 1.
        if graphics_format == GraphicsFormat.UNKNOWN: dtype = np.float32

    # Normalize UNORM
    if graphics_format & GraphicsFormat.UNORM: 
        im = im.clamp(-1., 1.) * 0.5 + 0.5

    im = (im * fac).cpu().numpy()

    if dtype not in [np.float32, np.float16]: im = im.clip(0, fac) # clamp uints before setting dtype
    im = im.astype(dtype).transpose((1, 2, 0))
    if file_fmt == ImageFileFormat.PNG: 
        # Pillow fucks up 16-bit PNGs
        iio.imwrite(fp, im, extension=fnext, plugin="PNG-FI")
    else:
        iio.imwrite(fp, im, extension=fnext)
    return True

def truncate_image(im:torch.Tensor) -> torch.Tensor:
    """
    Returns the first three color channels of a multi-channel image. If the tensor is sized [N=1, C, H, W], 
    a [C=3, H, W] tensor will be returned.

    :param torch.Tensor im: Multi-channel image tensor.
    :return: 3-channel image tensor.
    :rtype: torch.Tensor
    """
    imc = im.clone()
    if imc.dim() == 4 and size(0) == 1: imc = imc.squeeze(0)
    assert imc.dim() == 3, "image tensor has invalid size; expected [C=3+, H, W] or [N=1, C=3+, H, W]"
    assert imc.size(0) >= 3, "image tensor much have 3 or more color channels"
    return imc[0:3,...]