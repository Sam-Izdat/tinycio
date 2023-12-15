Image to CUBE LUT
=================

Create a color grading LUT by aligning the appearance of a source image to that of a target image.

.. It may be an interesting experiment (and fairly easy to implement) to add an option/flag for scanning 
.. directories and loading in multiple images, so that the LUT can be "trained" on several source and 
.. target files, in order to hopefully generalize better.

.. rubric:: Summary:

.. highlight:: text
.. code-block:: text

   usage: img2cube.py [-h] [--save-lut SAVE_LUT] [--save-image SAVE_IMAGE] [--size SIZE]
                      [--steps STEPS] [--learning-rate LEARNING_RATE] [--loss-fm LOSS_FM]
                      [--loss-jsd LOSS_JSD] [--loss-ssim LOSS_SSIM] [--empty-lut] [--igfs ]
                      [--igft ] [--ogf ] [--device ]
                      source target

   Apply an automatic color grade to an image and/or generate a color grading CUBE LUT
   by aligning the look of a source image to that of a target image.

   positional arguments:
     source                Source image file path
     target                Target image file path

   optional arguments:
     -h, --help            show this help message and exit
     --save-image SAVE_IMAGE, -i SAVE_IMAGE
                           Output image file path
     --save-lut SAVE_LUT, -l SAVE_LUT
                           Output LUT file path
     --size SIZE, -s SIZE  LUT size (range [0, 128]) (default: 64)
     --steps STEPS, -t STEPS
                           Steps (range [0, 10000]) (default: 1000)
     --learning-rate LEARNING_RATE, -r LEARNING_RATE
                           Learning rate (range [0, 1]) (default: 0.003)
     --strength STRENGTH
                           Strength of the effect (range [0, 1]) (default: 1.0)
     --empty-lut           Initialize empty LUT (instead of linear)
     --igfs []             Source image graphics format (default: unknown)
                           CHOICES:
                               sfloat16, sfloat32
                               uint8, uint16, uint32
     --igft []             Target image graphics format (default: unknown)
                           CHOICES: [same as above]
     --ogf []              Output image graphics format (default: unknown)
                           CHOICES: [same as above]
     --device []           Device for gradient descent (default: cuda)


.. rubric:: Example usage:

.. highlight:: shell
.. code-block:: shell

   $ ./img2cube.py --t 500 --save-image out.png source.png target.png

.. rubric:: Script:

.. literalinclude:: ../../scripts/img2cube.py
   :language: python