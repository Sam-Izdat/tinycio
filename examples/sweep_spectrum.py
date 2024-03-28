import torch
from tinycio import Color, ColorImage, Spectral

res = []
width = 800
for x in range(width):
    wl = (x / width) * 400 + 380
    res.append(Color(Spectral.wl_to_srgb_linear(wl, normalize=False)).image())

out = ColorImage(torch.cat(res, dim = 2).repeat(1, 50, 1), 'SRGB_LIN').clamp(0., 1.)
out *= 0.5
out.to_color_space('SRGB').save('../out/sweep_spectrum.png')