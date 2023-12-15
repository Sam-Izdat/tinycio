Hello Color
===========

A basic example:

.. highlight:: python
.. code-block:: python
    
    from tinycio import TransferFunction, fsio

    try:
        # Linearize an image
        im_srgb = fsio.load_image('my/srgb_image.png')
        im_linear = TransferFunction.srgb_eotf(im_srgb)
        fsio.save_image(im_linear, 'my/linear_image.png')
    except Exception as e: 
        print(e) # your error handling here

This package is divided into four user modules:

* :py:mod:`tinycio` - for all main color-related features and types
* :py:mod:`tinycio.fsio` - for loading from and saving to the file system
* :py:mod:`tinycio.util` - for miscellaneous color and non-color utility functions
* :py:mod:`tinycio.numerics` - for base numeric vector data types

It also exposes a few (optional) high-level abstractions to cut out a lot of the tedium:

* :class:`.ColorImage` - 3-channel *float32* color image type
* :class:`.Color` - 3-component *float32* color type
* :class:`.Chromaticity` - 2-component *float32* CIE xy chromaticity type

Exceptions may occur at any point, but exception handling will be omitted from 
subsequent code snippets for brevity.