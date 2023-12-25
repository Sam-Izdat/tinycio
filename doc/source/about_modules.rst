Modules
=======

The list of user-facing modules is just what you see :ref:`here <modindex>` and in :ref:`API Reference <api>`.

.. only:: html
	
	A list of *internal* modules (containing some user-facing code) can be found `here <../_modules/index.html>`_. 

.. note::

	The internal modules are not meant to be directly imported, unless you are modifying or developing the library. As they are not part of the user API, their names may change at any time. Import from their root module name instead - e.g. 

	* :code:`from tinycio.fsio import GraphicsFormat` - like this
	* :code:`from tinycio.fsio.format import GraphicsFormat` - not like this