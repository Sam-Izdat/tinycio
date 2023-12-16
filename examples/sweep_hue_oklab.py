import torch
from tinycio import Color, ColorImage
from tinycio.util import apply_hue_oklab

res = []
width = 800
for x in range(width):
    im = Color(1, 0, 0).image('SRGB_LIN').to_color_space('OKLAB')
    res.append(apply_hue_oklab(im, x / width * 2. - 1))

out = ColorImage(torch.cat(res, dim = 2).repeat(1,50,1), 'OKLAB')
out.to_color_space('SRGB').save('../out/sweep_hue_oklab.png')