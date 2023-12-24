=================================
Color spaces and graphics formats
=================================

.. _ref_color_spaces:

Color spaces
============

.. list-table:: Available color spaces
    :widths: 15 85
    :header-rows: 1

    *   - Identifier
        - Description
    *   - UNKNOWN
        - no color space specified - flag for "take a guess"
    *   - NONCOLOR
        - non-color data or color space not applicable
    *   - CIE_XYZ
        - CIE 1931 XYZ color space
    *   - CIE_XYY
        - xyY color space derived from above
    *   - SRGB
        - sRGB with gamma curve
    *   - SRGB_LIN
        - scene-linear sRGB/Rec. 709 - no gamma curve   
    *   - REC709
        - almost the same as sRGB, different gamma
    *   - REC2020
        - a wide-gamut RGB color space
    *   - REC2020_LIN
        - scene-linear Rec. 2020
    *   - DCI_P3
        - a wide-gamut RGB color space
    *   - DCI_P3_LIN    
        - scene-linear DCI-P3
    *   - DISPLAY_P3
        - some apple DCI P3 clone with different gamma, because apple
    *   - ACESCG
        - AP1 primaries; scene-linear, less suitable for color grading
    *   - ACESCC
        - AP1 primaries; logarithmic, more suitable for color grading
    *   - ACESCCT [1]_
        - AP1 primaries; logarithmic, more suitable for color grading 
    *   - ACES2065_1 [2]_
        - AP0 primaries; scene-linear, core ACES color space                        
    *   - LMS
        - "long, medium, short" / tristimulus
    *   - OKLAB
        - perceptual; Björn Ottosson's improvement over CIELAB
    *   - CIELAB
        - perceptual; for "characterization of colored surfaces and dyes"
    *   - CIELUV [3]_
        - perceptual; for "characterization of color displays"
    *   - HSV
        - "hue, saturation, value"; assumed sRGB-relative
    *   - HSL
        - "hue, saturation, lightness"; assumed sRGB-relative
    *   - OKHSV [4]_
        - perceptually linear HSV based on OKLAB
    *   - OKHSL [4]_
        - perceptually linear HSL based on OKLAB

.. rubric:: Footnotes

.. [1] not yet implemented
.. [2] scene-linear but still bad for rendering - can have negative values
.. [3] currently disabled
.. [4] intended for color picking; not implemented for image tensor conversion

.. _ref_graphics_formats:

Graphics formats
================

.. list-table:: Available graphics formats
    :widths: 15 85
    :header-rows: 1

    *   - Identifier
        - Description
    *   - UNKNOWN
        - Unknown; "take a guess"
    *   - UINT8
        - Unsigned 8-bit integers
    *   - UINT16
        - Unsigned 16-bit integers
    *   - UINT32
        - Unsigned 32-bit integers
    *   - SFLOAT16
        - Signed 16-bit floats
    *   - SFLOAT32
        - Signed 32-bit floats
    *   - UNORM8
        - Normalized unsigned 8-bit integers
    *   - UNORM16
        - Normalized unsigned 16-bit integers
    *   - UNORM32
        - Normalized unsigned 32-bit integers