import torch
from tinycio import Color, ColorImage

res = []
width = 800
col = Color(1, 1, 0.5)
for x in range(width):
    col.x = (x / width + 0.5) % 1
    res.append(col.image())

out = ColorImage(torch.cat(res, dim = 2).repeat(1, 50, 1).clamp(0., 1.), 'HSL')
out.to_color_space('SRGB').save('../out/sweep_hue_hsl.png')