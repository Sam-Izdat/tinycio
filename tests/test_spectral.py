import unittest
import numpy as np
from tinycio import Spectral


class TestSpectral(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cm_table_exceeds_one(self):
        # CIE 1931 2-degree CMFs legitimately exceed 1.0;
        # documents why wl_to_xyz must not clip to [0, 1].
        tab = Spectral.cm_table()
        self.assertGreater(float(tab[:, 0].max()), 1.0)   # x-bar peaks at 1.0622 (600nm)
        self.assertGreater(float(tab[:, 2].max()), 1.5)   # z-bar peaks at 1.7826 (445nm)
        self.assertAlmostEqual(float(tab[:, 0].max()), 1.0622, places=4)
        self.assertAlmostEqual(float(tab[:, 2].max()), 1.7826, places=4)

    def test_wl_to_xyz_no_clip(self):
        # Regression test: wl_to_xyz must return raw CMF values,
        # not clipped to [0, 1]. Spot-check wavelengths where Z/X exceed 1.
        for wl, expected in [(445., (0.3481, 0.0298, 1.7826)),
                             (450., (0.3362, 0.0380, 1.7721)),
                             (600., (1.0622, 0.6310, 0.0008))]:
            np.testing.assert_allclose(np.asarray(Spectral.wl_to_xyz(wl)),
                                       np.asarray(expected), atol=1e-4)

    def test_wl_to_xyz_chromaticity_450(self):
        # Clipping Z 1.7721 -> 1.0 shifts x 0.1566 -> 0.2447; guard the ratio.
        x, y, z = [float(v) for v in Spectral.wl_to_xyz(450.)]
        self.assertAlmostEqual(x / (x + y + z), 0.1566, places=4)

    def test_wl_to_xyz_matches_table_on_grid(self):
        # Every 5nm grid point must round-trip the embedded table exactly.
        tab = Spectral.cm_table()
        for i in range(tab.shape[0]):
            wl = 380. + 5. * i
            if wl >= 780:
                break
            np.testing.assert_allclose(np.asarray(Spectral.wl_to_xyz(wl)),
                                       np.asarray(tab[i]), atol=1e-6)

    def test_wl_to_xyz_interpolation_midpoint(self):
        # Off-grid wavelengths linearly interpolate between neighbors.
        tab = Spectral.cm_table()
        i = int((445. - 380.) / 5.)
        expected = 0.5 * (np.asarray(tab[i]) + np.asarray(tab[i + 1]))
        np.testing.assert_allclose(np.asarray(Spectral.wl_to_xyz(447.5)),
                                   expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
