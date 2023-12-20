White balance
=============

White balance an image.

.. rubric:: Summary:

.. highlight:: text
.. code-block:: text

   usage: tcio-white-balance [-h] --source-white SOURCE_WHITE --target-white TARGET_WHITE
                           [--color-space] [--igf ] [--ogf ]
                           input output

   White balance an image.

   positional arguments:
     input                 Input image file path
     output                Output image file path

   optional arguments:
     -h, --help            show this help message and exit
     --source-white SOURCE_WHITE, -s SOURCE_WHITE
                           Source white point - can be:
                            - string "auto"
                            - name of standard illuminant as e.g. "DAYLIGHT_NOON"
                            - correlated color temperature as e.g. 4600
                            - chromaticity xy as e.g. "0.123, 0.321"
     --target-white TARGET_WHITE, -t TARGET_WHITE
                           Target white point [as above, except no "auto"]
     --color-space , -c    Color space of input and output (default: srgb)
                           CHOICES:
                               cie_xyz, cie_xyy, srgb, srgb_linear
                               rec709, rec2020, dci_p3, display_p3
                               acescg, aces2065_1, lms, hsl, hsv
                               oklab, cielab
     --igf []              Input graphics format (default: unknown)
                           CHOICES:
                               sfloat16, sfloat32
                               uint8, uint16, uint32
     --ogf []              Output graphics format (default: unknown)
                           CHOICES: [same as above]

.. rubric:: Example usage:

.. highlight:: shell
.. code-block:: shell

   $ tcio-white-balance -s auto -t EQUAL_ENERGY -c srgb_linear env_map.exr env_map_bal.exr

.. rubric:: Script:

.. literalinclude:: ../../src/tinycio/scripts/white_balance.py
   :language: python