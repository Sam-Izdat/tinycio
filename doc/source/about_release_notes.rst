Release notes
=============
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