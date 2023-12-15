import torch
from tinycio import Chromaticity, ColorImage, WhiteBalance

res = []
width = 800
for x in range(width):
    cct = x / width * 21000 + 4000
    xy = Chromaticity(WhiteBalance.wp_from_cct(cct))
    col = xy.to_xyz(0.75)
    res.append(col.image())

out = ColorImage(torch.cat(res, dim = 2).repeat(1, 50, 1), 'CIE_XYZ')
out.to_color_space('SRGB').clamp(0., 1.).save('../out/sweep_cct.png')