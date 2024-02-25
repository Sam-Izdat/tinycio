Look up a color value
=====================

The tersest way:

.. highlight:: python
.. code-block:: python

    from tinycio import Color

    # Convert color from linear sRGB to CIE xyY
    result = Color(0.8, 0.4, 0.2).convert('SRGB_LIN', 'CIE_XYY')
    # returns Color([0.4129, 0.3817, 0.4706])

----

Slightly more verbose:

.. highlight:: python
.. code-block:: python

    from tinycio import Color 

    # Specify the color and make a 1x1 linear sRGB ColorImage
    col_im = Color(0.8, 0.4, 0.2).image('SRGB_LIN')

    # Convert sRGB linear image to CIE xyY
    result = Color(col_im.to_color_space('CIE_XYY'))
    # returns Color([0.4129, 0.3817, 0.4706])

This is functionally equivalent to:

.. highlight:: python
.. code-block:: python

    import torch
    from tinycio import ColorSpace

    # Specify the color as a tensor
    col_im = torch.tensor([0.8, 0.4, 0.2], dtype=torch.float32)

    # Expand it to a [3, 1, 1] sized image tensor
    col_im = col_im.unsqueeze(-1).unsqueeze(-1)

    # Convert
    cs_in = ColorSpace.Variant.SRGB_LIN
    cs_out = ColorSpace.Variant.CIE_XYY
    result = ColorSpace.convert(col_im, source=cs_in, destination=cs_out)
    # returns tensor([0.4129, 0.3817, 0.4706]) (squeezed)

.. note::
    Most color spaces are implemented as operations on image tensors, but a few 
    (OKHSL, OKHSV) are only available as direct color value conversions.

See: :py:meth:`.Color.convert`, :class:`.ColorSpace`