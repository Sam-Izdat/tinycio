Color to color
==============

Image-to-image color space conversion and tone mapping. 

.. rubric:: Summary:

.. highlight:: text
.. code-block:: text

   usage: tcio-color2color [-h] [--igf ] [--ogf ] [--tonemapper ] [--keep-alpha]
                         input input-color-space output output-color-space

   Convert the color space an image, with an optional tone mapping stage.

   positional arguments:
     input                 Input image file path
     input-color-space     Input color space
                           CHOICES:
                               cie_xyz, cie_xyy, srgb, srgb_linear
                               rec709, rec2020, dci_p3, display_p3
                               acescg, aces2065_1, lms, hsl, hsv
                               oklab, cielab
     output                Output image file path
     output-color-space    Output color space
                           CHOICES: [same as above]

   optional arguments:
     -h, --help            show this help message and exit
     --igf []              Input graphics format (default: unknown)
                           CHOICES:
                               sfloat16, sfloat32
                               uint8, uint16, uint32
     --ogf []              Output graphics format (default: unknown)
                           CHOICES: [same as above]
     --tonemapper [], -t []
                           Tone mapper (default: none)
                           CHOICES:
                               none, clamp, agx, agx_punchy
                               acescg, hable, reinhard
     --keep-alpha, -a      Preserve alpha channel


.. rubric:: Example usage:

.. highlight:: shell
.. code-block:: shell

   $ tcio-color2color -t agx input.tif cie_xyz output.png srgb

More explicit example:

.. highlight:: shell
.. code-block:: shell

   $ tcio-color2color --igf uint8 --ogf sfloat16 --keep-alpha input.png srgb output.tif aces2065_1

Also, note that the ACEScg RRT+ODT will not be applied automatically, so in this case "acescg" needs to be repeated - once for the "output color space" and once for the "tonemapper" option:

.. highlight:: shell
.. code-block:: shell

   $ tcio-color2color -t acescg input.exr acescg output.png srgb

.. rubric:: Script:

.. literalinclude:: ../../src/tinycio/scripts/color2color.py
   :language: python