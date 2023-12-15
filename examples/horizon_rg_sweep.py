import torch
import time
import numpy as np
from tinycio import Color, ColorImage

# Keying into PyTorch tensors for each pixel is prohibitively expensive.
# We can instead hand it over to NumPy - still slow, but relatively tolerable.
im_in = ColorImage.load('../doc/images/horizon.png', color_space='SRGB')
start = time.time()
im_in = im_in.to_color_space('SRGB_LIN').numpy()
C, H, W = im_in.shape
for y in range(H):
    for x in range(W):
        col = Color(im_in[:, y, x])
        col.r *= 1. - (x / W)
        col.g *= 1. - (y / H)
        im_in[:, y, x] = col.rgb
im_out = ColorImage(im_in, 'SRGB_LIN').to_color_space('SRGB').clamp(0., 1.)
end = time.time()
im_out.save('../out/horizon_rg_sweep.png')
print(f'Code execution: {end - start} seconds')