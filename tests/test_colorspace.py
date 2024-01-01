import unittest
import torch
import numpy as np
from tinycio import ColorSpace, Color

class TestColorSpaces(unittest.TestCase):
    # TODO: need to set tol per CS at some point...
    # rtol is not necessarily a good idea, as the tolerance is 
    # independent of any specific color value
    err_tol_gt_srgb = 0.005
    err_tol_self_ldr = 0.002
    err_tol_self_hdr = 0.025 
    err_tol_lab = 0.5 # LAB/LUV color can have huge values

    err_round = 3

    print_all = False

    lab = ColorSpace.Variant.CIELAB | ColorSpace.Variant.CIELUV
    cs_tests_color_class = [
            ColorSpace.Variant.SRGB, 
            ColorSpace.Variant.SRGB_LIN, 
            ColorSpace.Variant.CIE_XYZ, 
            ColorSpace.Variant.REC709, 
            ColorSpace.Variant.REC2020, 
            ColorSpace.Variant.REC2020_LIN, 
            ColorSpace.Variant.DCI_P3, 
            ColorSpace.Variant.DCI_P3_LIN, 
            ColorSpace.Variant.DISPLAY_P3, 
            ColorSpace.Variant.ACESCG, 
            ColorSpace.Variant.ACESCC, 
            ColorSpace.Variant.ACES2065_1, 
            ColorSpace.Variant.LMS, 
            ColorSpace.Variant.HSL, 
            ColorSpace.Variant.HSV, 
            ColorSpace.Variant.OKHSL, 
            ColorSpace.Variant.OKHSV, 
            ColorSpace.Variant.OKLAB, 
            ColorSpace.Variant.CIELAB]
    cs_tests_nocyl = [
            ColorSpace.Variant.SRGB, 
            ColorSpace.Variant.SRGB_LIN, 
            ColorSpace.Variant.CIE_XYZ, 
            ColorSpace.Variant.REC709, 
            ColorSpace.Variant.REC2020, 
            ColorSpace.Variant.REC2020_LIN,
            ColorSpace.Variant.DCI_P3, 
            ColorSpace.Variant.DCI_P3_LIN, 
            ColorSpace.Variant.DISPLAY_P3, 
            ColorSpace.Variant.ACESCG, 
            ColorSpace.Variant.ACESCC, 
            ColorSpace.Variant.ACES2065_1, 
            ColorSpace.Variant.LMS, 
            ColorSpace.Variant.OKLAB, 
            ColorSpace.Variant.CIELAB]
    cs_tests_all = [
            ColorSpace.Variant.SRGB, 
            ColorSpace.Variant.SRGB_LIN, 
            ColorSpace.Variant.CIE_XYZ, 
            ColorSpace.Variant.REC709, 
            ColorSpace.Variant.REC2020, 
            ColorSpace.Variant.DCI_P3, 
            ColorSpace.Variant.DISPLAY_P3, 
            ColorSpace.Variant.ACESCG, 
            ColorSpace.Variant.ACESCC, 
            ColorSpace.Variant.ACES2065_1, 
            ColorSpace.Variant.LMS, 
            ColorSpace.Variant.HSL, 
            ColorSpace.Variant.HSV, 
            ColorSpace.Variant.OKLAB, 
            ColorSpace.Variant.CIELAB]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check_diff_np(self, v1, v2, tol):
        return np.abs(v1 - v2).max() < tol

    def check_diff_torch(self, v1, v2, tol):
        return torch.allclose(
            v1,
            v2,
            atol=tol)

    check_diff = check_diff_torch

    def test_gt_color_class(self):
        testname = "GROUND TRUTH"
        # GT values from https://ajalt.github.io/colormath/converter/
        # not sure how accurate the conversions are, but it's close enough 
        # to see if something is obviously broken
        #
        # TODO: add more GT tests
        cs = ColorSpace.Variant
        col_srgb_lin = Color(0.2, 0.4, 0.7) # sRGB linear
        gt_vals = {
          cs.SRGB:          Color(0.48453, 0.66519, 0.85431),
          cs.CIE_XYZ:       Color(0.35185, 0.37913, 0.71692),
          cs.REC709:        Color(0.43367, 0.62865, 0.83703),
          cs.HSL:           Color(0.58524, 0.55928, 0.66942),
          cs.HSV:           Color(0.58524, 0.43284, 0.85431),
          cs.CIELAB:        Color(67.95644, -2.86733, -29.22882),
          cs.OKLAB:         Color(0.72224, -0.02924, -0.08135),
          cs.ACES2065_1:    Color(0.36504, 0.41079, 0.65867),
          cs.ACESCC:        Color(0.45318, 0.47717, 0.52029),
          cs.ACESCG:        Color(0.29111, 0.38958, 0.65773),
          cs.REC2020:       Color(0.52805, 0.61997, 0.81587),
          cs.DCI_P3:        Color(0.56384, 0.69821, 0.85856),
          cs.DISPLAY_P3:    Color(0.52255, 0.66018, 0.83772)
        }

        for k, v in gt_vals.items():
            res = col_srgb_lin.convert(cs.SRGB_LIN, k)
            compare = self.check_diff_np(res, v, self.err_tol_lab if k & self.lab else self.err_tol_gt_srgb)
            if not compare or self.print_all:
                print("===========")
                print(testname)
                print("ERROR: ", np.abs(Color(v)-Color(res)).round(decimals=self.err_round))
                print("sRGB seed val", Color(col_srgb_lin))
                print(
                    Color(v).round(decimals=2),"vs", Color(res).round(decimals=2),
                    "OK" if compare else "**", cs.SRGB_LIN.name, "<->", k.name)
                print("===========")
            self.assertTrue(compare)

    def test_out_of_gamut(self):
        testname = "OUT OF GAMUT"

        cs = ColorSpace.Variant
        # TODO: Needs more...
        oog_vals = {
          cs.CIE_XYZ:          Color(0.1, 0.8, 0.2)
        }

        for k, v in oog_vals.items():
            orig = v.image(k)
            res = orig.to_color_space(cs.SRGB).to_color_space(k)
            compare = self.check_diff(orig, res, self.err_tol_self_ldr)
            if not compare or self.print_all:
                print("===========")
                print(testname)
                print("ERROR: ", np.abs(Color(orig)-Color(res)).round(decimals=self.err_round))
                print(
                    Color(v).round(decimals=2),"vs", Color(res).round(decimals=2),
                    "OK" if compare else "**", k.name, "<->", cs.SRGB.name)
                print("===========")


    def test_self_consistency_color_class(self):
        testname = "SELF CONSISTENCY - COLOR CLASS"
        # Tolerance set relatively high 
        # Cylindrical transformations HSV/HSL omitted
        cols_srgb = [
            Color(1., 0., 0.), 
            Color(0., 1., 0.), 
            Color(0., 0., 1.), 
            Color(1., 1., 1.), 
            Color(0.25, 0.45, 0.85)]
        cs_tests = self.cs_tests_nocyl

        col_out, col_back = None, None
        np.set_printoptions(suppress=True)
        for col_srgb in cols_srgb:
            for cs_in in cs_tests:
                for cs_test in cs_tests:
                    if cs_in == cs_test: continue
                    col_in = col_srgb.convert(ColorSpace.Variant.SRGB_LIN, cs_in)
                    col_out = col_in.convert(cs_in, cs_test)
                    col_back = col_out.convert(cs_test, cs_in)
                    compare = self.check_diff_np(col_in, col_back, self.err_tol_lab if cs_in & self.lab else self.err_tol_self_ldr)
                    if not compare or self.print_all:
                        print("===========")
                        print(testname)
                        print("ERROR: ", np.abs(Color(col_in)-Color(col_back)).round(decimals=self.err_round))
                        print("sRGB seed val", Color(col_srgb))
                        print(
                            Color(col_in).round(decimals=2),"vs", Color(col_back).round(decimals=2), "by", 
                            Color(col_out.round(decimals=2)), 
                            "OK" if compare else "**", cs_in.name, "<->", cs_test.name)
                        print("===========")
                    self.assertTrue(compare)

    def test_self_consistency_ldr(self):
        testname = "LDR SELF CONSISTENCY"
        # Tolerance set relatively high 
        # Cylindrical transformations HSV/HSL omitted
        cols_srgb = [
            Color(1., 0., 0.).image(), 
            Color(0., 1., 0.).image(), 
            Color(0., 0., 1.).image(), 
            Color(1., 1., 1.).image(), 
            Color(0.25, 0.45, 0.85).image()]
        cs_tests = self.cs_tests_nocyl

        col_out, col_back = None, None
        np.set_printoptions(suppress=True)
        for col_srgb in cols_srgb:
            for cs_in in cs_tests:
                for cs_test in cs_tests:
                    if cs_in == cs_test: continue
                    col_in = ColorSpace.convert(col_srgb, ColorSpace.Variant.SRGB_LIN, cs_in)
                    col_out = ColorSpace.convert(col_in, cs_in, cs_test)
                    col_back = ColorSpace.convert(col_out, cs_test, cs_in)
                    compare = self.check_diff(col_in, col_back, self.err_tol_lab if cs_in & self.lab else self.err_tol_self_ldr)
                    if not compare or self.print_all:
                        print("===========")
                        print(testname)
                        print("ERROR: ", np.abs(Color(col_in)-Color(col_back)).round(decimals=self.err_round))
                        print("sRGB seed val", Color(col_srgb))
                        print(
                            Color(col_in).round(decimals=2),"vs", Color(col_back).round(decimals=2), "by", 
                            Color(col_out.round(decimals=2)), 
                            "OK" if compare else "**", cs_in.name, "<->", cs_test.name)
                        print("===========")
                    self.assertTrue(compare)

    def test_self_consistency_hdr(self):
        testname = "HDR SELF CONSISTENCY"
        # Tolerance set relatively high
        # Cylindrical transformations HSV/HSL omitted - they need LDR inputs
        cols_srgb = [
            Color(10., 0., 0.).image(), 
            Color(0., 10., 0.).image(), 
            Color(0., 0., 10.).image(), 
            Color(10., 10., 10.).image(), 
            Color(10.25, 20.45, 30.85).image()]
        cs_tests = self.cs_tests_nocyl

        col_out, col_back = None, None
        np.set_printoptions(suppress=True)
        for col_srgb in cols_srgb:
            for cs_in in cs_tests:
                for cs_test in cs_tests:
                    if cs_in == cs_test: continue
                    col_in = ColorSpace.convert(col_srgb, ColorSpace.Variant.SRGB_LIN, cs_in)
                    col_out = ColorSpace.convert(col_in, cs_in, cs_test)
                    col_back = ColorSpace.convert(col_out, cs_test, cs_in)
                    compare = self.check_diff(col_in, col_back, self.err_tol_lab if cs_in & self.lab else self.err_tol_self_hdr)
                    if not compare or self.print_all:
                        print("===========")
                        print(testname)
                        print("ERROR: ", np.abs(Color(col_in)-Color(col_back)).round(decimals=self.err_round))
                        print("sRGB seed val", Color(col_srgb))
                        print(
                            Color(col_in).round(decimals=2),"vs", Color(col_back).round(decimals=2), "by", 
                            Color(col_out.round(decimals=2)), 
                            "OK" if compare else "**", cs_in.name, "<->", cs_test.name)
                        print("===========")
                    self.assertTrue(compare)

    def test_srgb_consistency_ldr(self):
        testname = "LDR SRGB CONSISTENCY"
        # Testing with lower tolerance
        cols_in = [
            Color(1., 0., 0.).image(), 
            Color(0., 1., 0.).image(), 
            Color(0., 0., 1.).image(), 
            Color(1., 1., 1.).image(), 
            Color(0.25, 0.45, 0.85).image()]
        cs_tests = self.cs_tests_all

        col_out, col_back = None, None
        np.set_printoptions(suppress=True)
        for col_in in cols_in:
            for cs_in in [ColorSpace.Variant.SRGB_LIN]:
                for cs_test in cs_tests:
                    if cs_in == cs_test or self.print_all: continue
                    col_out = ColorSpace.convert(col_in, cs_in, cs_test)
                    col_back = ColorSpace.convert(col_out, cs_test, cs_in)
                    compare = self.check_diff(col_in, col_back, self.err_tol_lab if cs_in & self.lab else self.err_tol_self_ldr)
                    if not compare:
                        print("===========")
                        print(testname)
                        print("ERROR: ", np.abs(Color(col_in)-Color(col_back)).round(decimals=self.err_round))
                        print(
                            Color(col_in).round(decimals=3),"vs", Color(col_back).round(decimals=3), "by", 
                            Color(col_out.round(decimals=3)), 
                            "OK" if compare else "**", cs_in.name, "<->", cs_test.name)
                        print("===========")
                    self.assertTrue(compare)

    def test_srgb_consistency_hdr(self):
        testname = "HDR SRGB CONSISTENCY"
        # Testing with lower tolerance
        # Cylindrical transformations HSV/HSL omitted - they need LDR inputs
        cols_in = [
            Color(10., 0., 0.).image(), 
            Color(0., 10., 0.).image(), 
            Color(0., 0., 10.).image(), 
            Color(10., 10., 10.).image(), 
            Color(10.25, 20.45, 30.85).image()]
        
        cs_tests = self.cs_tests_nocyl

        col_out, col_back = None, None
        np.set_printoptions(suppress=True)
        for col_in in cols_in:
            for cs_in in [ColorSpace.Variant.SRGB_LIN]:
                for cs_test in cs_tests or self.print_all:
                    if cs_in == cs_test: continue
                    col_out = ColorSpace.convert(col_in, cs_in, cs_test)
                    col_back = ColorSpace.convert(col_out, cs_test, cs_in)
                    compare = self.check_diff(col_in, col_back, self.err_tol_lab if cs_in & self.lab else self.err_tol_self_hdr)
                    if not compare:
                        print("===========")
                        print(testname)
                        print("ERROR: ", np.abs(Color(col_in)-Color(col_back)).round(decimals=self.err_round))
                        print(
                            Color(col_in),"vs", Color(col_back), "by", 
                            Color(col_out.round(decimals=3)), 
                            "OK" if compare else "**", cs_in.name, "<->", cs_test.name)
                        print("===========")
                    self.assertTrue(compare)

if __name__ == '__main__':
    unittest.main()