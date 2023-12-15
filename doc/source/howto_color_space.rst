Change an image's color space
=============================

.. highlight:: python
.. code-block:: python

    from tinycio import ColorImage

    # Color space is assumed to be sRGB, bit depth assumed to be uint8 in, float32 out
    ColorImage.load('my/image.png').to_color_space('ACES2065_1').save('my/image.exr')

----

A little more explicit:

.. highlight:: python
.. code-block:: python

    from tinycio import ColorImage

    # Load sRGB image from disk
    im_in = ColorImage.load('my/image.png', graphics_format='R8G8B8_UINT', color_space='SRGB')

    # Convert the image
    im_out = im_in.to_color_space('ACES2065_1')

    # Save as 32-bit-per-channel EXR file
    im_out.save('my/image.exr', graphics_format='R32G32B32_SFLOAT')

This is functionally equivalent to:

.. highlight:: python
.. code-block:: python

    from tinycio import ColorSpace, fsio

    # Load image - returns [C, H, W] sized torch.Tensor
    im_in = fsio.load_image('my/image.png')

    # Specify color spaces
    cs_in = ColorSpace.Variant.SRGB
    cs_out = ColorSpace.Variant.ACES2065_1

    # Covert the image
    im_out = ColorSpace.convert(im_in, cs_in, cs_out)

    # Save as 32-bit-per-channel EXR file
    fmt_out = fsio.GraphicsFormat.R32G32B32_SFLOAT
    fsio.save_image(im_out, 'my/image.exr', graphics_format=fmt_out)

.. note:: 
    Tone mapping is treated as a discrete step and, in the general case, 
    converted color values will not be normalized or clamped to a valid range 
    for the destination color space automatically. 

See: :py:meth:`.ColorImage.to_color_space`, :class:`.ColorSpace`