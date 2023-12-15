import torch
import time
from tinycio import ColorImage, ColorCorrection

# Each vertical line of the image hue shifted as a tensor.
res = []
im_in = ColorImage.load('../doc/images/horizon.png', color_space='SRGB')
start = time.time()
im_in = im_in.to_color_space('OKLAB')
cc = ColorCorrection()
_, H, W = im_in.size()
for x in range(W): 
	cc.set_hue_delta(x / W * 2. - 1)
	res.append(im_in[:, :, x:x+1].correct(cc))
im_out = ColorImage(torch.cat(res, dim = 2), 'OKLAB')
im_out = im_out.to_color_space('SRGB').clamp(0., 1.)
end = time.time()
im_out.save('../out/horizon_hue_sweep.png')
print(f'Code execution: {end - start} seconds')