![tinycio](https://raw.githubusercontent.com/Sam-Izdat/tinycio/main/doc/images/tinycio_sm.png)

* [Project site & docs](https://sam-izdat.github.io/tinycio-docs/) 
* [PyPi](https://pypi.org/project/tinycio/)

# About

A primitive, lightweight Python color library for PyTorch-involved projects. It implements color space conversion, tone mapping, LUT usage and creation, basic color correction and color balancing, and HDR-LDR encoding/decoding. 

# Getting started

* Recommended: set up a clean Python environment
* [Install PyTorch  as instructed here](https://pytorch.org/get-started/locally/)
* Run  `pip install tinycio`
* Run  `tcio-setup` ([iio docs on fi](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.freeimage.html#module-imageio.plugins.freeimage))

[See the docs](https://sam-izdat.github.io/tinycio-docs/) for the rest.

# Requires

- PyTorch >=2.0 (earlier versions untested)
- NumPy >=1.21
- imageio >=2.9 (with PNG-FI FreeImage plugin)
- tqdm >=4.64
- toml >=0.10