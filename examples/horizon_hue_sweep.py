import torch
import time
from tinycio import ColorImage
from tinycio.util import apply_hue_oklab

# Each vertical line of the image hue shifted as a tensor.
res = []
im_in = ColorImage.load('../doc/images/examples_ll/horizon.png', 'SRGB')
start = time.time()
im_in = im_in.to_color_space('OKLAB')
_, H, W = im_in.size()
for x in range(W): 
	res.append(apply_hue_oklab(im_in[:, :, x:x+1], x / W * 2. - 1))
im_out = ColorImage(torch.cat(res, dim = 2).clamp(0., 1.), 'OKLAB')
im_out = im_out.to_color_space('SRGB')
end = time.time()
im_out.save('../out/horizon_hue_sweep.png')
print(f'Code execution: {end - start} seconds')