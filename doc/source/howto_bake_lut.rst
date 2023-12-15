Bake color correction to a LUT
==============================

.. highlight:: python
.. code-block:: python
	
	from tinycio import ColorCorrection, 

	cc = ColorCorrection()
	cc.set_saturation(1.5)
	cc.set_contrast(1.2)
	cc.set_hue_delta(0.1)
	lut = cc.bake_lut()
	lut.save('my/lut.cube')

----

Keep in mind that, by default, the LUT will be baked for the color grading color space – *ACEScc*. 
So, before using that LUT, you would need to put the image in its color space.

.. highlight:: python
.. code-block:: python
	
	from tinycio import ColorImage, LookupTable

	lut = LookupTable.load('my/lut.cube')
	im = ColorImage.load('my/image.exr', 'SRGB_LIN')
	im_cc = im.to_color_space('ACESCC')
	im_out = im_cc.lut(lut)
	im_out.to_color_space('SRGB_LIN').save('my/corrected_image.exr')

Alternatively, you can bake the LUT in the color space of your preference:

.. highlight:: python
.. code-block:: python
	
	from tinycio import ColorCorrection

	cc = ColorCorrection()
	# [...apply color correction...]
	lut = cc.bake_lut(lut_size=64, lut_color_space='SRGB')
	lut.save('my/lut_srgb_lut.cube')

See: :py:meth:`.ColorCorrection.bake_lut`