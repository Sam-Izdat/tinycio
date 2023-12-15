import unittest
import numpy as np

from tinycio import ColorCorrection

class TestCodec(unittest.TestCase):
	
	def setUp(self):
		pass

	def tearDown(self):
		pass


	def test_load_save(self):
		cc1 = ColorCorrection()
		cc1.set_exposure_bias(-2)
		cc1.set_color_filter(0.5,0.5)
		cc1.set_hue_delta(-0.33)
		cc1.set_saturation(0.25)
		cc1.set_contrast(1.5)
		cc1.set_shadow_color(0.1,0.2)
		cc1.set_midtone_color(0.2,0.3)
		cc1.set_highlight_color(0.3,0.4)
		cc1.set_shadow_offset(-1.2)
		cc1.set_midtone_offset(-1.5)
		cc1.set_highlight_offset(-1.7)
		cc1.save('../out/cc_test.toml')
		cc2 = ColorCorrection.load('../out/cc_test.toml')

		self.assertEqual(cc1.exposure_bias, cc2.exposure_bias)
		self.assertTrue(np.array_equal(cc1.color_filter, cc2.color_filter))
		self.assertEqual(cc1.hue_delta, cc2.hue_delta)
		self.assertEqual(cc1.saturation, cc2.saturation)
		self.assertEqual(cc1.contrast, cc2.contrast)
		self.assertTrue(np.array_equal(cc1.shadow_color, cc2.shadow_color))
		self.assertTrue(np.array_equal(cc1.midtone_color, cc2.midtone_color))
		self.assertTrue(np.array_equal(cc1.highlight_color, cc2.highlight_color))
		self.assertEqual(cc1.shadow_offset, cc2.shadow_offset)
		self.assertEqual(cc1.midtone_offset, cc2.midtone_offset)
		self.assertEqual(cc1.highlight_offset, cc2.highlight_offset)

if __name__ == '__main__':
	unittest.main()