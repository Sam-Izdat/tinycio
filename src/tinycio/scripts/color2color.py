#!/usr/bin/env python
"""Convert the color space an image, with an optional tone mapping stage."""
import os
import argparse
from argparse import RawTextHelpFormatter
import torch
from tinycio import ColorImage, ColorSpace, ToneMapping, fsio

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument(
        'ics', 
        type=str, 
        choices=[
            'cie_xyz', 'cie_xyy', 'srgb', 'srgb_lin',
            'rec709', 'rec2020', 'rec2020_lin', 
            'dci_p3', 'dci_p3_lin', 'display_p3',
            'acescg', 'aces2065_1', 'acescc', 'lms', 
            'hsl', 'hsv', 'oklab', 'cielab'],
        metavar='input-color-space', # because bad formatting is bad :(
        help='Input color space\n' + \
            'CHOICES:\n' + \
            '    cie_xyz, cie_xyy, srgb, srgb_lin, \n' + \
            '    rec709, rec2020, rec_2020_lin, \n' + \
            '    dci_p3, dci_p3_lin, display_p3, \n' + \
            '    acescg, acescc, aces2065_1, \n' + \
            '    lms, hsl, hsv, oklab, cielab'
        )
    parser.add_argument('output', type=str, help='Output image file path')
    parser.add_argument(
        'ocs',
        type=str, 
        choices=[
            'cie_xyz', 'cie_xyy', 'srgb', 'srgb_lin',
            'rec709', 'rec2020', 'rec2020_lin', 
            'dci_p3', 'dci_p3_lin', 'display_p3',
            'acescg', 'aces2065_1', 'acescc', 'lms', 
            'hsl', 'hsv', 'oklab', 'cielab'],
        metavar='output-color-space',
        help='Output color space\n' + \
            'CHOICES: [same as above]'  
        )
    parser.add_argument(
        '--tmcs', 
        type=str, 
        default="srgb_lin", 
        const="srgb_lin",
        nargs='?',
        choices=['srgb_lin', 'rec2020_lin', 'dci_p3_lin', 'acescg', 'aces2065_1'],
        metavar='', 
        help='Tone mapping color space (default: %(default)s)\n' + \
            'CHOICES:\n' + \
            '    srgb_lin, rec2020_lin \n' + \
            '    dci_p3_lin, acescg, aces2065_1 \n'
        )
    parser.add_argument(
        '--igf', 
        type=str, 
        default="unknown", 
        const="unknown",
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
        default="unknown", 
        const="unknown",
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='',
        help='Output graphics format (default: %(default)s)\n' + \
            'CHOICES: [same as above]'
        )
    parser.add_argument(
        '--tonemapper',
        '-t',
        type=str, 
        default="none",
        const="unknown",
        nargs='?',
        choices=[
            'none', 'clamp', 'agx', 'agx_punchy',
            'acescg', 'hable', 'reinhard'],
        metavar='',
        help='Tone mapper (default: %(default)s)\n' + \
            'CHOICES:\n' + \
            '    none, clamp, agx, agx_punchy \n' + \
            '    acescg, hable, reinhard'
        )
    parser.add_argument('--keep-alpha', '-a', action='store_true', help='Preserve alpha channel')
    args = parser.parse_args()

    fp_in   = os.path.realpath(args.input)
    fp_out  = os.path.realpath(args.output)
    gf_in   = args.igf.strip().upper()
    gf_out  = args.ogf.strip().upper()
    cs_in   = args.ics.strip().upper()
    cs_out  = args.ocs.strip().upper()
    tm      = args.tonemapper.strip().upper()
    alpha   = args.keep_alpha

    try:
        im, ac = None, None
        if alpha:
            im = fsio.load_image(fp_in, graphics_format=fsio.GraphicsFormat[gf_in])
            ac = im[3:4,...] if im.size(0) == 4 else None
            im = ColorImage(fsio.truncate_image(im), color_space=cs_in)
        else:
            im = ColorImage.load(fp_in, color_space=cs_in, graphics_format=gf_in)

        im = im.tone_map(tm, target_color_space=args.tmcs) # it's okay to pass "NONE"
        im = im.to_color_space(cs_out)
        if ac is not None: im = torch.cat([im, ac], dim=0)
        im.save(fp_out, gf_out)
        print(f'saved image to: {os.path.realpath(fp_out)}')
    except Exception as e: 
        print(f'cannot convert: {e}') 
        
if __name__ == '__main__':
    main_cli()