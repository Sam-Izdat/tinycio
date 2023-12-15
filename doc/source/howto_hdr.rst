Encode HDR data as RGBA PNG
===========================

Formats like Logluv and RGBE can encode three channels of HDR data as a 32-bpp RGBA image.

.. highlight:: python
.. code-block:: python

    from tinycio import fsio, Codec

    # Load 3-channel HDR image
    im_hdr = fsio.load_image('my/hdr_image.exr')

    # Encode logluv - returns 4-channel tensor
    im_logluv = Codec.logluv_encode(im_hdr)

    # Save 4-channel LogLuv image
    fsio.save_image(im_logluv, 'my/logluv_image.png')

.. highlight:: python
.. code-block:: python

    from tinycio import fsio, Codec

    # Load 4-channel LogLuv image
    im_logluv = fsio.load_image('my/logluv_image.png')

    # Decode LogLuv - returns 3-channel tensor
    im_hdr = Codec.logluv_decode(im_logluv)

    # Save 3-channel HDR image
    fsio.save_image(im_hdr, 'my/hdr_image.exr')

.. note::
    :class:`tinycio.Codec` doesn't care about your color space and you can feed it any HDR data you like. 
    You can encode a :class:`tinycio.ColorImage`, but the effect will be the same as any other image tensor.

See: :class:`.Codec`