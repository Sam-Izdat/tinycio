import torch
from tinycio import Color, ColorImage

res = []
width = 800
for x in range(width):
    res.append(Color(1, 0, 0).image('SRGB_LIN').apply_hue(x / width * 2. - 1))

out = ColorImage(torch.cat(res, dim = 2).repeat(1,50,1), 'SRGB_LIN').clamp(0., 1.)
out.to_color_space('SRGB').save('../out/sweep_hue_oklab.png')