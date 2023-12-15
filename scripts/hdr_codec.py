#!/usr/bin/env python
"""Encode and decode high dynamic range color data."""
import os
import argparse
from argparse import RawTextHelpFormatter
from tinycio import fsio, Codec

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input', type=str, help='Input image file path')
parser.add_argument('output', type=str, help='Output image file path')
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--encode', '-e', action='store_true', help='Encode mode')
mode.add_argument('--decode', '-d', action='store_true', help='Decode mode')
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
    '--format', 
    type=str, 
    choices=['logluv'],
    default='logluv',
    help='Format'
    )
args = parser.parse_args()

fp_in   = os.path.realpath(args.input)
fp_out  = os.path.realpath(args.output)
gf_in   = args.igf.strip().upper()
gf_out  = args.ogf.strip().upper()

try:
    if args.format == 'logluv':
        if args.encode:
            im_hdr = fsio.load_image(fp_in, graphics_format=fsio.GraphicsFormat[gf_in])
            im_hdr = fsio.truncate_image(im_hdr)
            im_logluv = Codec.logluv_encode(im_hdr)
            fsio.save_image(im_logluv, fp_out) 
        elif args.decode:
            im_logluv = fsio.load_image(fp_in, graphics_format=fsio.GraphicsFormat[gf_in])
            im_hdr = Codec.logluv_decode(im_logluv)
            fsio.save_image(im_hdr, fp_out)
        else:
            raise Exception('unexpected mode')
    else:
        raise Exception('unexpected format')
    print(f'saved image to: {fp_out}')
except Exception as e: 
    print(f'cannot encode/decode: {e}')