import unittest
import warnings
import torch
import numpy as np
from tinycio import *
from tinycio.numerics import *

class TestCodec(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_tutorial(self):
        # Writes ~10MB into out directory
        # Only tests that the tutorial code is error-free; output will be mostly nonsense
        testval = True

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            test_fp = '../doc/images/test_image.png'

            # ---------------------------------------------------------------------

            # Linearize an image
            im_srgb = fsio.load_image(test_fp)
            im_linear = TransferFunction.srgb_eotf(im_srgb)
            fsio.save_image(im_linear, '../out/1.png')

            # ---------------------------------------------------------------------

            # Look up a wavelength (nm) as CIE XYZ color
            col_xyz = Spectral.wl_to_xyz(490)   # Float3([0.0320, 0.2080, 0.4652])

            # Look up a wavelength as sRGB color
            col_srgb = Spectral.wl_to_srgb(490) # Float3([0.0320, 0.2080, 0.4652])

            # ---------------------------------------------------------------------

            # Convert color from linear sRGB to CIE XYZ
            result = Color(0.8, 0.4, 0.2).convert('SRGB_LIN', 'CIE_XYY')

            # Specify the color and make a 1x1 linear sRGB ColorImage
            col_im = Color(0.8, 0.4, 0.2).image('SRGB_LIN')

            # Convert sRGB linear image to CIE xyY
            result = Color(col_im.to_color_space('CIE_XYY'))
            # returns Color([0.4129, 0.3817, 0.4706])


            # Specify the color as a tensor
            col_im = torch.tensor([0.8, 0.4, 0.2], dtype=torch.float32)

            # Expand it to a [3, 1, 1] sized image tensor
            col_im = col_im.unsqueeze(-1).unsqueeze(-1)

            # Convert
            cs_in = ColorSpace.Variant.SRGB_LIN
            cs_out = ColorSpace.Variant.CIE_XYY
            result = ColorSpace.convert(col_im, source=cs_in, destination=cs_out)
            # returns tensor([0.4129, 0.3817, 0.4706])

            # ---------------------------------------------------------------------

            # Color space is assumed to be sRGB, bit depth assumed to be uint8 in, float32 out
            ColorImage.load(test_fp).to_color_space('ACES2065_1').save('../out/4a.exr')


            # Load sRGB image from disk
            im_in = ColorImage.load(test_fp, graphics_format='UINT8', color_space='SRGB')

            # Convert the image
            im_out = im_in.to_color_space('ACES2065_1')

            # Save as 32-bit-per-channel EXR file
            im_out.save('../out/4b.exr', graphics_format='SFLOAT32')


            # Load image - returns [C, H, W] sized torch.Tensor
            im_in = fsio.load_image(test_fp)

            # Specify color spaces
            cs_in = ColorSpace.Variant.SRGB
            cs_out = ColorSpace.Variant.ACES2065_1

            # Covert the image
            im_out = ColorSpace.convert(im_in, cs_in, cs_out)

            # Save as 32-bit-per-channel EXR file
            fmt_out = fsio.GraphicsFormat.SFLOAT32
            fsio.save_image(im_out, '../out/4c.exr', graphics_format=fmt_out)

            # ---------------------------------------------------------------------

            im = ColorImage.load(test_fp, 'SRGB_LIN')
            cc = ColorCorrection()
            cc.set_contrast(1.2)
            cc.set_saturation(0.8)
            im.correct(cc).save('../out/5a.exr')

            im_in = fsio.load_image(test_fp)
            cs_in = ColorSpace.Variant.SRGB_LIN
            cs_cc = ColorSpace.Variant.ACESCC
            im_cc = ColorSpace.convert(im_in, cs_in, cs_cc)
            cc = ColorCorrection()
            cc.set_contrast(1.2)
            cc.set_saturation(1.2)
            im_corrected = cc.apply(im_cc)
            fsio.save_image(im_corrected, '../out/5b.exr')

            # ---------------------------------------------------------------------

            lut = LookupTable.get_negative()
            ColorImage.load(test_fp).lut(lut).save('../out/6a.png')


            # Load and linearize
            im = ColorImage.load(test_fp, 'SRGB').to_color_space('SRGB_LIN')

            # Apply desired curve
            im_log_c = TransferFunction.log_c_oetf(im)

            # Apply LUT
            im_log_c = lut.apply(im_log_c)

            # Back to linear (only if needed - a display-ready LUT should have the transform baked-in)
            im_out = TransferFunction.log_c_eotf(im)

            # Apply sRGB gamma curve (if needed) and save
            ColorImage(im_out, 'SRGB_LIN').to_color_space('SRGB').save('../out/6b.png')

            # ---------------------------------------------------------------------

            im = ColorImage.load(test_fp, 'SRGB_LIN')
            im = im.white_balance(source_white='auto', target_white='HORIZON')
            im.save('../out/7a.tif', graphics_format='SFLOAT16')


            cs_lin = ColorSpace.Variant.SRGB_LIN
            cs_xyz = ColorSpace.Variant.CIE_XYZ
            cs_lms = ColorSpace.Variant.LMS
            wp_target = WhiteBalance.Illuminant.HORIZON
            fmt_out = fsio.GraphicsFormat.SFLOAT16

            # Load the image
            im = fsio.load_image(test_fp)

            # Convert to XYZ for white point estimation
            im_xyz = ColorSpace.convert(im, cs_lin, cs_xyz)

            # Get approximate source white point
            src_white = WhiteBalance.wp_from_image(im_xyz)

            # Get xy chromaticity from standard illuminant
            tgt_white = WhiteBalance.wp_from_illuminant(wp_target)

            # Convert image to LMS for white balancing
            im_lms = ColorSpace.convert(im, cs_lin, cs_lms)

            # Apply
            im_lms = WhiteBalance.apply(im_lms, source_white=src_white, target_white=tgt_white)

            # Convert back to sRGB linear
            im_out = ColorSpace.convert(im, cs_lms, cs_lin)

            # Finally save
            fsio.save_image(im_out, '../out/7.tif', graphics_format=fmt_out)

            # ---------------------------------------------------------------------


            # Load 3-channel HDR image
            im_hdr = fsio.load_image(test_fp)

            # Encode logluv - returns 4-channel tensor
            im_logluv = Codec.logluv_encode(im_hdr)

            # Save 4-channel LogLuv image
            fsio.save_image(im_logluv, '../out/8.png')

            # Load 4-channel LogLuv image
            im_logluv = fsio.load_image('../out/8.png')

            # Decode LogLuv - returns 3-channel tensor
            im_hdr = Codec.logluv_decode(im_logluv)

            # Save 3-channel HDR image
            fsio.save_image(im_hdr, '../out/8b.png')

        self.assertTrue(testval)

if __name__ == '__main__':
    unittest.main()