Release notes
=============

.. rubric:: v 0.7.2 a - Apr. 2024

* Fixed ACES RRT/ODT

.. rubric:: v 0.7.1 a - Apr. 2024

* Misc doc and script corrections

.. rubric:: v 0.7.0 a - Apr. 2024

* Added :code:`target_color_space` argument to :class:`.ColorImage` tone mapping. Image will be returned and, if possible, tone mapped in the desired color space.
* Added ACEScc color space to :code:`tcio-color2color` script

.. rubric:: v 0.6.3 a - Mar. 2024

* Minor correction to sRGB transfer functions (insignificant for 8-bit values)
* Added luminance scaling for spectral wavelength-to-sRGB functions

.. rubric:: v 0.6.2 a - Dec. 2023

* Fixed color space conversion bugs resulting from clipping values
* Fixed color correction bug: midtone saturation was being assigned wrong values
* Added better transforms for REC2020/REC2020_LIN and DCI_P3/DCI_P3_LIN - fixed DCI chromatic adaptation

.. rubric:: v 0.6.1 a - Dec. 2023

* Corrected linear color space definitions for scripts
* Updated numerics unit test

.. rubric:: v 0.6.0 a - Dec. 2023

* **API change**: simplified :class:`~.fsio.GraphicsFormat` (options now just UINT8, SFLOAT32, UNORM8, etc).
* Added denormalization/normalization for UNORM image loading/saving.
* API extension: extended numeric types to accept and return lists and tuples.
* Moved scripts to src and added them to setuptools config. Scripts can now be run with:

	* :code:`tcio-color2color`
	* :code:`tcio-hdr-codec`
	* :code:`tcio-img2cube`
	* :code:`tcio-white-balance`

* Added :code:`tcio-setup` script. Currently just an alias for :code:`imageio_download_bin freeimage`.

.. rubric:: v 0.5.4 a - Dec. 2023

* Fixed tonemapping color space conversion bug with :class:`.ColorImage`.

.. rubric:: v 0.5.3 a - Dec. 2023

* Updated examples to reflect API changes.
* Fixed potential problem with ColorImage prematurely losing state.
* Updated docs.

.. rubric:: v 0.5.2 a - Dec. 2023

* Very minor corrections and fixes, mostly to docs and examples.

.. rubric:: v 0.5.1 a - Dec. 2023

* just docs and webhosting housekeeping

.. rubric:: v 0.5.0 a - Dec. 2023

* *gestures at the docs*

----

Fin.