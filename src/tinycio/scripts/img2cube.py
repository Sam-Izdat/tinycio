#!/usr/bin/env python
"""
Apply an automatic color grade to an image and/or generate a color grading CUBE LUT 
by aligning the look of a source image to that of a target image.
"""
import os
import argparse
from argparse import RawTextHelpFormatter
from tinycio import fsio, LookupTable
from tinycio.util import progress_bar

def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source', type=str, help='Source image file path')
    parser.add_argument('target', type=str, help='Target image file path')
    parser.add_argument('--save-image', '-i', type=str, help='Output image file path')
    parser.add_argument('--save-lut', '-l', type=str, help='Output LUT file path')
    parser.add_argument('--size', '-s', 
        type=int, 
        default=64, 
        help='LUT size (range [0, 128]) (default: %(default)s)')
    parser.add_argument('--steps', '-t', 
        type=int, 
        default=1000, 
        help='Steps (range [0, 10000]) (default: %(default)s)')
    parser.add_argument('--learning-rate', '-r', 
        type=float, 
        default=0.003,
        help='Learning rate (range [0, 1]) (default: %(default)s)')
    parser.add_argument('--strength', 
        type=float, 
        default=1.,
        help='Strength of the effect (range [0, 1]) (default: %(default)s)')
    parser.add_argument('--empty-lut', action='store_true', help='Initialize empty LUT (instead of linear)')
    parser.add_argument(
        '--igfs', 
        type=str, 
        default="unknown", 
        const="unknown",
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='', 
        help='Source image graphics format (default: %(default)s)\n' + \
            'CHOICES:\n' + \
            '    sfloat16, sfloat32 \n' + \
            '    uint8, uint16, uint32 \n'
        )
    parser.add_argument(
        '--igft',
        type=str, 
        default="unknown", 
        const="unknown",
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='',
        help='Target image graphics format (default: %(default)s)\n' + \
            'CHOICES: [same as above]'
        )
    parser.add_argument(
        '--ogf',
        type=str, 
        default="unknown", 
        const="unknown",
        nargs='?',
        choices=['sfloat16','sfloat32','uint8','uint16','uint32'],
        metavar='',
        help='Output image graphics format (default: %(default)s)\n' + \
            'CHOICES: [same as above]'
        )
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda", 
        const="cuda",
        nargs='?',
        choices=['cpu','cuda'],
        metavar='', 
        help='Device for gradient descent (default: %(default)s)\n'
        )
    args = parser.parse_args()

    try:
        if not (args.save_lut or args.save_image):
            parser.error('no output requested - need at least one of: --save-lut or --save-image')

        assert 8. <= args.size <= 128, "size must be in range [0, 128]"
        assert 1 <= args.steps <= 10000, "steps must be in range [0, 10000]"
        assert 0. <= args.strength <= 1., "strength must be in range [0, 1]"
        assert 0. <= args.learning_rate <= 1., "learning-rate must be in range [0, 1]"

        steps   = int(args.steps)
        fp_src  = os.path.realpath(args.source) if args.source else None
        fp_tgt  = os.path.realpath(args.target) if args.target else None
        fp_out  = os.path.realpath(args.save_image) if args.save_image else None
        fp_lut  = os.path.realpath(args.save_lut) if args.save_lut else None
        gfs_in  = args.igfs.strip().upper()
        gft_in  = args.igft.strip().upper()
        gf_out  = args.ogf.strip().upper()

        im_src = fsio.load_image(fp_src, graphics_format=fsio.GraphicsFormat[gfs_in])
        im_dst = fsio.load_image(fp_tgt, graphics_format=fsio.GraphicsFormat[gft_in])
        im_src = fsio.truncate_image(im_src)
        im_dst = fsio.truncate_image(im_dst)

        lut = LookupTable.get_empty(args.size) if args.empty_lut else LookupTable.get_linear(args.size) 
        lut.fit_to_image(
            im_source=im_src, 
            im_target=im_dst, 
            strength=args.strength,
            steps=steps,
            device=args.device,
            context=progress_bar
            )

        im_out = lut.apply(im_src)
        if fp_lut: lut.save(fp_lut)
        if fp_out: fsio.save_image(im_out, fp_out, graphics_format=fsio.GraphicsFormat[gf_out])
    except Exception as e: 
        print(f'cannot proceed: {e}')
        
if __name__ == '__main__':
    main_cli()