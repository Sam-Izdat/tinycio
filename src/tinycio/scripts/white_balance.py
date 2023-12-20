#!/usr/bin/env python
"""White balance an image."""
import os
import argparse
from argparse import RawTextHelpFormatter
from tinycio import ColorImage, Chromaticity

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path')
    parser.add_argument(
        '--source-white',
        '-s',
        required=True,
        type=str,
        help='Source white point - can be:\n' + \
            ' - string "auto"\n' + \
            ' - name of standard illuminant as e.g. "DAYLIGHT_NOON"\n' + \
            ' - correlated color temperature as e.g. 4600\n' + \
            ' - chromaticity xy as e.g. "0.123, 0.321"\n'
        )
    parser.add_argument(
        '--target-white', 
        '-t',
        required=True,
        type=str,
        help='Target white point [as above, except no "auto"]'
        )
    parser.add_argument(
        '--color-space',
        '-c',
        type=str, 
        default='srgb',
        metavar='',
        choices=[
            'cie_xyz', 'cie_xyy', 'srgb', 'srgb_lin',
            'rec709', 'rec2020', 'rec2020_lin', 
            'dci_p3', 'dci_p3_lin', 'display_p3',
            'acescg', 'aces2065_1', 'lms', 'hsl', 'hsv',
            'oklab', 'cielab'],
        help='Input color space\n' + \
            'CHOICES:\n' + \
            '    cie_xyz, cie_xyy, srgb, srgb_lin, \n' + \
            '    rec709, rec2020, rec_2020_lin, \n' + \
            '    dci_p3, dci_p3_lin, display_p3, \n' + \
            '    acescg, aces2065_1, lms, hsl, hsv \n' + \
            '    oklab, cielab'
        )
    parser.add_argument(
        '--igf', 
        type=str, 
        default='unknown', 
        const='unknown',
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='', 
        help='Input graphics format (default: %(default)s)\n' + \
            'CHOICES:\n' + \
            '    sfloat16, sfloat32 \n' + \
            '    uint8, uint16, uint32 \n'
        )
    parser.add_argument(
        '--ogf',
        type=str, 
        default='unknown', 
        const='unknown',
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='',
        help='Output graphics format (default: %(default)s)\n' + \
            'CHOICES: [same as above]'
        )
    args = parser.parse_args()

    fp_in       = os.path.realpath(args.input)
    fp_out      = os.path.realpath(args.output)
    cs          = args.color_space.strip().upper()
    gf_in       = args.igf.strip().upper()
    gf_out      = args.ogf.strip().upper()

    src_white   = None
    tgt_white   = None

    try:
        if "," in args.source_white:
            src_white = args.source_white.split(',')
            if len(src_white) == 2:
                src_white = Chromaticity(float(src_white[0]), float(src_white[1]))
            else:
                raise Exception('script could not interpret source white')
        else:
            src_white = args.source_white.strip()
            if src_white.isnumeric(): src_white = int(src_white)

        if "," in args.target_white:
            tgt_white = args.target_white.split(',')
            if len(tgt_white) == 2:
                tgt_white = Chromaticity(float(tgt_white[0]), float(tgt_white[1]))
            else:
                raise Exception('script could not interpret target white')
        else:
            tgt_white = args.target_white.strip()
            if tgt_white.isnumeric(): tgt_white = int(tgt_white)

        im = ColorImage.load(fp_in, color_space=cs, graphics_format=gf_in)
        im = im.white_balance(src_white, tgt_white)
        im.save(fp_out, graphics_format=gf_out)
    except Exception as e: 
        print(f'cannot apply: {e}') 

if __name__ == '__main__':
    main_cli()