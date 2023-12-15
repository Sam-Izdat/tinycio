Lower-level image manipulation
==============================

It is generally far better to operate on whole tensors rather than individual pixels, because reading and writing per-pixel or even per-line data is very slow with native Python. But you can make it work in a pinch. If you need actual performance, take a look at `Taichi Lang <https://www.taichi-lang.org/>`_ or `Numba <https://numba.pydata.org/>`_. 

Source image
------------

.. image:: ../images/examples_ll/horizon.png
    :width: 700
    :alt: OKLAB hue sweep

Line sweep
----------

.. literalinclude:: ../../examples/horizon_hue_sweep.py
   :language: python

.. figure:: ../images/examples_ll/horizon_hue_sweep.png
   :width: 700
   :alt: Image line sweep

   \~0.2s, depending on hardware - not great.

Pixel sweep
-----------

.. literalinclude:: ../../examples/horizon_rg_sweep.py
   :language: python

.. figure:: ../images/examples_ll/horizon_rg_sweep.png
   :width: 700
   :alt: Image pixel sweep

   \~4.5s - getting yikesy

Baseline
--------

.. literalinclude:: ../../examples/horizon_rg_torch.py
   :language: python

Same operation, but with PyTorch functions: ~0.05s