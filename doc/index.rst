tinycio
=======================================

Primitive, lightweight Python color library for PyTorch-involved projects. It implements color space conversion, tone mapping, LUT usage and creation, basic color correction and color balancing, and HDR-LDR encoding/decoding. 

.. uncomment for PDF build ("only" directive is ignored):

.. .. toctree::
..     :hidden:

..     self

Getting started
---------------

#. Recommended: set up a clean Python environment
#. `Install PyTorch  as instructed here <https://pytorch.org/get-started/locally/>`_
#. Run :code:`pip install tinycio`
#. Run :code:`tcio-setup` (installs freeimage binaries; `iio docs on fi <https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.freeimage.html#module-imageio.plugins.freeimage>`_)

About 
-----

.. only:: latex

    Version 
    |release|

.. only:: html

    .. hlist::
        :columns: 4
    
        * v |release|
        * `PDF manual <./tinycio.pdf>`_
        * `Previous versions <https://github.com/Sam-Izdat/tinycio/releases>`_
        * :doc:`Release notes <source/about_release_notes>`

As pip has a tendency to screw up PyTorch installations, torch is deliberately left out of 
the explicit package requirements; install it whichever way is most appropriate, if you 
haven't done so already.

.. rubric:: Note on limitations

I feel like a disclaimer is in order.

* This package is (obviously) not suitable for realtime rendering
* This package is *generally* not suitable for differentiable rendering
* I am not a color expert and can't guarantee scientific accuracy

This package was motivated by a frustration with the staggering complexities of digital color and put together 
as part of a larger project, already using PyTorch, to minimize external dependencies. In self-flattering terms, 
the goal here is a kind of spartan-utilitarian approach to color for image processing with some light machine learning. 
In less self-flattering terms, I have no idea what I'm doing. This is not meant to be a replacement for a proper 
color management solution.

While the code was tested within reason, some things are likely still not quite correct. For dependable color management, 
`defer to OCIO <https://opencolorio.org/>`_ or consider a more comprehensive and scientifically-minded Python 
project like `Colour <https://www.colour-science.org/>`_.

.. rubric:: Supported color spaces (and other things treated as if they were):

* CIE XYZ
* CIE xyY
* sRGB
* Rec. 709 
* sRGB / Rec. 709 (scene-linear) 
* Rec. 2020 
* Rec. 2020 (scene-linear) 
* DCI-P3 
* DCI-P3 (scene-linear) 
* Display P3
* ACEScg
* ACEScc
* ACES 2065-1
* LMS
* OKLAB
* CIELAB
* HSL
* HSV
* OKHSL
* OKHSV

.. rubric:: Supported tone mapping operators:

* AgX / AgX punchy
* ACEScg (fitted RRT+ODT)
* Hable (a.k.a. Uncharted 2)
* Reinhard (extended)

.. rubric:: Supported LUT formats:

* CUBE

.. rubric:: License

:doc:`MIT License <source/license>` on all original code - see source for details

How to
------

.. only:: html

    The :doc:`how-to section <source/howto_hello>` has a brief tutorial on the library's core functionality.

.. toctree::
    :maxdepth: 4
    :caption: How to:
    :hidden:

    source/howto_hello
    source/howto_wavelength
    source/howto_color_value
    source/howto_color_space
    source/howto_hdr
    source/howto_tone_map
    source/howto_white_balance
    source/howto_ccbasic
    source/howto_apply_lut
    source/howto_bake_lut

Examples
--------

.. only:: html

    See: :doc:`examples section <source/example_sweeps>`.

.. toctree::
    :maxdepth: 4
    :caption: Examples:
    :hidden:

    source/example_sweeps
    source/example_image_manip


Deeper dive
------------

.. only:: html

    See: :doc:`deeper dives section <source/deep_autograde>`.

.. only:: not latex

    .. toctree::
        :maxdepth: 4
        :caption: Deeper dive:
        :hidden:

        source/deep_autograde


Reference
---------

.. only:: html

    See: :doc:`reference section <source/tinycio>`.

.. toctree::
    :maxdepth: 2
    :caption: Reference:
    :hidden:

    source/tinycio
    source/ref_cs_gfx
    source/about_release_notes
    source/about_modules
    source/license
    genindex


Scripts
-------

.. only:: html

    The library comes with a few bundled CLI scripts:

    * :doc:`tcio-color2color <source/scripts_color2color>`
    * :doc:`tcio-hdr-codec <source/scripts_hdr_codec>`
    * :doc:`tcio-white-balance <source/scripts_white_balance>`
    * :doc:`tcio-img2cube <source/scripts_img2cube>`

.. toctree::
    :maxdepth: 2
    :caption: Scripts:
    :hidden:

    source/scripts_color2color
    source/scripts_hdr_codec
    source/scripts_white_balance
    source/scripts_img2cube

Links
-----

* `GitHub <https://github.com/Sam-Izdat/tinycio>`_
* `PyPi <https://pypi.org/project/tinycio/>`_

.. toctree::
    :maxdepth: 2
    :caption: Links:
    :hidden:

    GitHub <https://github.com/Sam-Izdat/tinycio>
    PyPi <https://pypi.org/project/tinycio/>
    Docs <https://sam-izdat.github.io/tinycio/>

Sibling projects
----------------

* `tinytex <https://sam-izdat.github.io/tinytex/>`_
* `tinyfilm <https://sam-izdat.github.io/tinyfilm/>`_

.. toctree::
    :maxdepth: 2
    :caption: Sibling projects:
    :hidden:

    tinytex <https://sam-izdat.github.io/tinytex/>
    tinyfilm <https://sam-izdat.github.io/tinyfilm/>

Special thanks
--------------

* Some color space conversion is from `S2CRNet <https://github.com/stefanLeong/S2CRNet>`_, `convert-colors-py <https://github.com/CairX/convert-colors-py>`_ and `seal-3d <https://github.com/windingwind/seal-3d>`_. 
* Several matrices were computed with `colorspace-routines <https://github.com/m13253/colorspace-routines>`_.
* CCT calculation is from `colour-science <https://github.com/colour-science/>`_.
* The AgX implementation is owed to `Troy James Sobotka <https://github.com/sobotka/AgX-S2O3>`_ and `Liam Collod <https://github.com/MrLixm/AgXc>`_. 
* Some loss computations are borrowed from `NLUT <https://github.com/semchan/NLUT/tree/main>`_. 
* Thanks to the `@64 blog <https://64.github.io/tonemapping/>`_ for explaining common tone mapping algorithms. 
* `Stephen Hill (@self_shadow) <https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl>`_ for ACES tone mapping fit.
* `John Hable <http://filmicworlds.com/blog/filmic-tonemapping-operators/>`_ for the Uncharted 2 operator.
* `Reinhard et al. <https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf>`_ for their operator. 
* The white balancing and the von Kries transform were kindly explained by `pbr-book <https://pbr-book.org/4ed/Cameras_and_Film/Film_and_Imaging>`_.
* The `OKLAB <https://bottosson.github.io/posts/oklab/>`_ color space was developed by `Björn Ottosson <https://bottosson.github.io/>`_
* The `OKHSL and OKHSV <https://bottosson.github.io/posts/colorpicker/>`_ color space conversions originally by  `Brian Holbrook <https://github.com/holbrookdev>`_
* Matt of Ready At Dawn Studios for `explaining Logluv <https://mynameismjp.wordpress.com/2008/12/12/logluv-encoding-for-hdr/>`_.
* `Test photograph <./_images/wb4k6k12k.jpg>`_ is from `Bianca Salgado <https://www.pexels.com/@biancasalgado/>`_