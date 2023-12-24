import unittest
import torch

from tinycio import ColorImage

class TestCodec(unittest.TestCase):
    
    err_tol = 0.01
    tm_list = ['AGX', 'AGX_PUNCHY', 'ACESCG', 'HABLE', 'REINHARD']

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check_diff_torch(self, v1, v2, tol):
        return torch.allclose(
            v1,
            v2,
            atol=tol)

    def test_cs(self):
        # Checks color space conversion consistency
        for tm in self.tm_list:
            im = ColorImage.load('../doc/images/test_image.png', 'SRGB')
            im1 = im.tone_map(tm)
            im2 = im.to_color_space('HSL').tone_map(tm).to_color_space('SRGB')
            compare = self.check_diff_torch(im1, im2, self.err_tol)
            self.assertTrue(compare)

if __name__ == '__main__':
    unittest.main()