import torch
import time
from tinycio import ColorImage

im_in = ColorImage.load('../doc/images/horizon.png', color_space='SRGB')
start = time.time()
im_in = im_in.to_color_space('SRGB_LIN')
C, H, W = im_in.shape
xw = torch.linspace(start=1., end=0., steps=W).unsqueeze(0).repeat(H, 1)
yh = torch.linspace(start=1., end=0., steps=H).unsqueeze(-1).repeat(1, W)
im_in[0,...] *= xw
im_in[1,...] *= yh
im_out = ColorImage(im_in, 'SRGB_LIN').to_color_space('SRGB').clamp(0., 1.)
end = time.time()
im_out.save('../out/horizon_rg_torch.png')
print(f'Code execution: {end - start} seconds')