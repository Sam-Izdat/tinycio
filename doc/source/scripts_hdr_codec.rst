HDR Codec
=========

Encode and decode HDR color data to and from low-bit-depth RGBA files.

.. rubric:: Summary:

.. highlight:: text
.. code-block:: text

   usage: tcio-hdr-codec [-h] (--encode | --decode) [--igf ] [--ogf ] [--format {logluv}] input output

   Encode and decode high dynamic range color data.

   positional arguments:
     input              Input image file path
     output             Output image file path

   optional arguments:
     -h, --help         show this help message and exit
     --encode, -e       Encode mode
     --decode, -d       Decode mode
     --igf []           Input graphics format (default: unknown)
                        CHOICES:
                            sfloat16, sfloat32
                            uint8, uint16, uint32
     --ogf []           Output graphics format (default: unknown)
                        CHOICES: [same as above]
     --format {logluv}  Format

.. rubric:: Example usage:

.. highlight:: shell
.. code-block:: shell

   $ tcio-hdr-codec --encode hdr_image.exr encoded_image.png

.. highlight:: shell
.. code-block:: shell

   $ tcio-hdr-codec --decode encoded_image.png hdr_image.exr

.. rubric:: Script:

.. literalinclude:: ../../src/tinycio/scripts/hdr_codec.py
   :language: python