Look up a wavelength
=====================

.. highlight:: python
.. code-block:: python

    from tinycio import Spectral

    # Look up a wavelength (nm) as CIE XYZ color
    col_xyz = Spectral.wl_to_xyz(490)   # Float3([0.0320, 0.2080, 0.4652])

    # Look up a wavelength as sRGB color
    col_srgb = Spectral.wl_to_srgb(490) # Float3([0.0320, 0.2080, 0.4652])

.. image:: ../images/howto_spectral/490nm.png
    :width: 25
    :alt: 510nm
    :align: center

See: :class:`.Spectral`