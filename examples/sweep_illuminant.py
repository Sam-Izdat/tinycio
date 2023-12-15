import torch
from tinycio import Chromaticity, ColorImage, WhiteBalance

res = []
width = 800
illum = WhiteBalance.Illuminant.TUNGSTEN
illum_xy = Chromaticity(WhiteBalance.wp_from_illuminant(illum))
light_scale, dist_scale = 2000., 1000.
for x in range(width):
    col = illum_xy.to_xyz(light_scale * (1./(((x / width) * dist_scale)**2 + 1.)))
    res.append(col.image())

out = ColorImage(torch.cat(res, dim = 2).repeat(1, 50, 1), 'CIE_XYZ')
out.to_color_space('SRGB').clamp(0., 1.).save('../out/sweep_illuminant.png')