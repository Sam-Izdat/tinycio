import unittest
import torch
import numpy as np
from tinycio.numerics import *
from tinycio import Color

class TestNumerics(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_val(self):
        self.assertTrue(np.array_equal(Float2(1.,2.), np.array([1.,2.])))
        self.assertTrue(np.array_equal(Float3(1.,2.,3.), np.array([1.,2.,3.])))
        self.assertTrue(np.array_equal(Float4(1.,2.,3.,4.), np.array([1.,2.,3.,4.])))

        self.assertTrue(np.array_equal(Float4([1.,2.,3.,4.]), np.array([1.,2.,3.,4.])))
        self.assertTrue(np.array_equal(Float4((1.,2.,3.,4.)), np.array([1.,2.,3.,4.])))

        self.assertTrue(np.array_equal(Int2(1,2), np.array([1,2])))
        self.assertTrue(np.array_equal(Int3(1,2,3), np.array([1,2,3])))
        self.assertTrue(np.array_equal(Int4(1,2,3,4), np.array([1,2,3,4])))

    def test_single_val(self):
        self.assertTrue(np.array_equal(Float2(1.), np.array([1.,1.])))
        self.assertTrue(np.array_equal(Float3(1.), np.array([1.,1.,1.])))
        self.assertTrue(np.array_equal(Float4(1.), np.array([1.,1.,1.,1.])))

        self.assertTrue(np.array_equal(Int2(1), np.array([1,1])))
        self.assertTrue(np.array_equal(Int3(1), np.array([1,1,1])))
        self.assertTrue(np.array_equal(Int4(1), np.array([1,1,1,1])))

    def test_np(self):
        self.assertTrue(np.array_equal(Float2(np.array([1.,2.])), np.array([1.,2.])))
        self.assertTrue(np.array_equal(Float3(np.array([1.,2.,3.])), np.array([1.,2.,3.])))
        self.assertTrue(np.array_equal(Float4(np.array([1.,2.,3.,4.])), np.array([1.,2.,3.,4.])))

        self.assertTrue(np.array_equal(Int2(np.array([1,2])), np.array([1,2])))
        self.assertTrue(np.array_equal(Int3(np.array([1,2,3])), np.array([1,2,3])))
        self.assertTrue(np.array_equal(Int4(np.array([1,2,3,4])), np.array([1,2,3,4])))

    def test_torch(self):
        self.assertTrue(np.array_equal(Float2(torch.Tensor([1.,2.])), np.array([1.,2.])))
        self.assertTrue(np.array_equal(Float3(torch.Tensor([1.,2.,3.])), np.array([1.,2.,3.])))
        self.assertTrue(np.array_equal(Float4(torch.Tensor([1.,2.,3.,4.])), np.array([1.,2.,3.,4.])))

        self.assertTrue(np.array_equal(Int2(torch.Tensor([1,2])), np.array([1,2])))
        self.assertTrue(np.array_equal(Int3(torch.Tensor([1,2,3])), np.array([1,2,3])))
        self.assertTrue(np.array_equal(Int4(torch.Tensor([1,2,3,4])), np.array([1,2,3,4])))

    def test_image_method(self):
        self.assertTrue(np.array_equal(Float3(1.), Float3(Color(1.).image())))
        self.assertTrue(np.array_equal(Float3(1.,2.,3.), Float3(Color(1.,2.,3.).image())))

    def test_swizzle(self):
        # Testing all would just be repeating the implementation, but let's cast a pretty wide net
        self.assertEqual(Float2(1,2).x, 1)
        self.assertEqual(Float2(1,2).y, 2)
        self.assertTrue(np.array_equal(Float2(1,2), Float2(1,2).xy))
        self.assertTrue(np.array_equal(Float2(1,2).yx, Float2(2,1)))
        self.assertTrue(np.array_equal(Float2(1,2).xx, Float2(1,1)))
        self.assertTrue(np.array_equal(Float2(1,2).yy, Float2(2,2)))
        self.assertTrue(np.array_equal(Float2(1,2).xxx, Float3(1)))
        self.assertTrue(np.array_equal(Float2(1,2).yyy, Float3(2)))
        self.assertTrue(np.array_equal(Float2(1,2).xxxx, Float4(1)))
        self.assertTrue(np.array_equal(Float2(1,2).yyyy, Float4(2)))
        self.assertTrue(np.array_equal(Float2(1,2).xyxy, Float4(1,2,1,2)))
        self.assertTrue(np.array_equal(Float2(1,2).yx, Float2(1,2).gr))


        self.assertEqual(Float3(1,2,3).x, 1)
        self.assertEqual(Float3(1,2,3).y, 2)
        self.assertEqual(Float3(1,2,3).z, 3)
        self.assertTrue(np.array_equal(Float3(1,2,3).xyz, Float3(1,2,3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).zyx, Float3(3,2,1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).xxx, Float3(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).yyy, Float3(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).zzz, Float3(3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).xy, Float2(1,2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).xx, Float2(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).yy, Float2(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).zz, Float2(3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).xxxx, Float4(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).yyyy, Float4(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).zzzz, Float4(3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).zyx, Float3(1,2,3).bgr))

        self.assertEqual(Float4(1,2,3,4).x, 1)
        self.assertEqual(Float4(1,2,3,4).y, 2)
        self.assertEqual(Float4(1,2,3,4).z, 3)
        self.assertEqual(Float4(1,2,3,4).w, 4)
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xyzw, Float4(1,2,3,4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xyzx, Float4(1,2,3,1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xxxx, Float4(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).yyyy, Float4(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).zzzz, Float4(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).wwww, Float4(4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xy, Float2(1,2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xx, Float2(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).yy, Float2(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).zz, Float2(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).ww, Float2(4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).xxx, Float3(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).yyy, Float3(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).zzz, Float3(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).www, Float3(4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).wzyx, Float4(1,2,3,4).abgr))

        self.assertEqual(Int2(1,2).x, 1)
        self.assertEqual(Int2(1,2).y, 2)
        self.assertTrue(np.array_equal(Int2(1,2), Int2(1,2).xy))
        self.assertTrue(np.array_equal(Int2(1,2).yx, Int2(2,1)))
        self.assertTrue(np.array_equal(Int2(1,2).xx, Int2(1,1)))
        self.assertTrue(np.array_equal(Int2(1,2).yy, Int2(2,2)))
        self.assertTrue(np.array_equal(Int2(1,2).xxx, Int3(1)))
        self.assertTrue(np.array_equal(Int2(1,2).yyy, Int3(2)))
        self.assertTrue(np.array_equal(Int2(1,2).xxxx, Int4(1)))
        self.assertTrue(np.array_equal(Int2(1,2).yyyy, Int4(2)))
        self.assertTrue(np.array_equal(Int2(1,2).xyxy, Int4(1,2,1,2)))
        self.assertTrue(np.array_equal(Int2(1,2).yx, Int2(1,2).gr))

        self.assertEqual(Int3(1,2,3).x, 1)
        self.assertEqual(Int3(1,2,3).y, 2)
        self.assertEqual(Int3(1,2,3).z, 3)
        self.assertTrue(np.array_equal(Int3(1,2,3).xyz, Int3(1,2,3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zyx, Int3(3,2,1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).xxx, Int3(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).yyy, Int3(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zzz, Int3(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).xy, Int2(1,2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).xx, Int2(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).yy, Int2(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zz, Int2(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).xxxx, Int4(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).yyyy, Int4(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zzzz, Int4(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zyx, Int3(1,2,3).bgr))

        self.assertEqual(Int4(1,2,3,4).x, 1)
        self.assertEqual(Int4(1,2,3,4).y, 2)
        self.assertEqual(Int4(1,2,3,4).z, 3)
        self.assertEqual(Int4(1,2,3,4).w, 4)
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xyzw, Int4(1,2,3,4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xyzx, Int4(1,2,3,1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).wzyx, Int4(4,3,2,1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xxxx, Int4(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).yyyy, Int4(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).zzzz, Int4(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).wwww, Int4(4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xy, Int2(1,2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xx, Int2(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).yy, Int2(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).zz, Int2(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).ww, Int2(4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).xxx, Int3(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).yyy, Int3(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).zzz, Int3(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).www, Int3(4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).wzyx, Int4(1,2,3,4).abgr))


        self.assertEqual(Float2(1,2).r, 1)
        self.assertEqual(Float2(1,2).g, 2)
        self.assertTrue(np.array_equal(Float2(1,2), Float2(1,2).rg))
        self.assertTrue(np.array_equal(Float2(1,2).gr, Float2(2,1)))
        self.assertTrue(np.array_equal(Float2(1,2).rr, Float2(1,1)))
        self.assertTrue(np.array_equal(Float2(1,2).gg, Float2(2,2)))
        self.assertTrue(np.array_equal(Float2(1,2).rrr, Float3(1)))
        self.assertTrue(np.array_equal(Float2(1,2).ggg, Float3(2)))
        self.assertTrue(np.array_equal(Float2(1,2).rrrr, Float4(1)))
        self.assertTrue(np.array_equal(Float2(1,2).gggg, Float4(2)))
        self.assertTrue(np.array_equal(Float2(1,2).rgrg, Float4(1,2,1,2)))

        self.assertEqual(Float3(1,2,3).r, 1)
        self.assertEqual(Float3(1,2,3).g, 2)
        self.assertEqual(Float3(1,2,3).b, 3)
        self.assertTrue(np.array_equal(Float3(1,2,3).rgb, Float3(1,2,3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).bgr, Float3(3,2,1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).rrr, Float3(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).ggg, Float3(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).rg, Float2(1,2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).rr, Float2(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).gg, Float2(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).bb, Float2(3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).bbb, Float3(3)))
        self.assertTrue(np.array_equal(Float3(1,2,3).rrrr, Float4(1)))
        self.assertTrue(np.array_equal(Float3(1,2,3).gggg, Float4(2)))
        self.assertTrue(np.array_equal(Float3(1,2,3).bbbb, Float4(3)))

        self.assertEqual(Float4(1,2,3,4).r, 1)
        self.assertEqual(Float4(1,2,3,4).g, 2)
        self.assertEqual(Float4(1,2,3,4).b, 3)
        self.assertEqual(Float4(1,2,3,4).a, 4)
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rgba, Float4(1,2,3,4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rgbr, Float4(1,2,3,1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rrrr, Float4(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).gggg, Float4(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).bbbb, Float4(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).aaaa, Float4(4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rg, Float2(1,2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rr, Float2(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).gg, Float2(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).bb, Float2(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).aa, Float2(4)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).rrr, Float3(1)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).ggg, Float3(2)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).bbb, Float3(3)))
        self.assertTrue(np.array_equal(Float4(1,2,3,4).aaa, Float3(4)))

        self.assertEqual(Float2(1,2).r, 1)
        self.assertEqual(Float2(1,2).g, 2)
        self.assertTrue(np.array_equal(Float2(1,2), Float2(1,2).rg))
        self.assertTrue(np.array_equal(Float2(1,2).gr, Float2(2,1)))
        self.assertTrue(np.array_equal(Float2(1,2).rr, Float2(1,1)))
        self.assertTrue(np.array_equal(Float2(1,2).gg, Float2(2,2)))
        self.assertTrue(np.array_equal(Float2(1,2).rrr, Float3(1)))
        self.assertTrue(np.array_equal(Float2(1,2).ggg, Float3(2)))
        self.assertTrue(np.array_equal(Float2(1,2).rrrr, Float4(1)))
        self.assertTrue(np.array_equal(Float2(1,2).gggg, Float4(2)))
        self.assertTrue(np.array_equal(Float2(1,2).rgrg, Float4(1,2,1,2)))

        self.assertEqual(Int3(1,2,3).r, 1)
        self.assertEqual(Int3(1,2,3).g, 2)
        self.assertEqual(Int3(1,2,3).b, 3)
        self.assertTrue(np.array_equal(Int3(1,2,3).rgb, Int3(1,2,3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).bgr, Int3(3,2,1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).rrr, Int3(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).ggg, Int3(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).bbb, Int3(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).rg, Int2(1,2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).rr, Int2(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).gg, Int2(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).bb, Int2(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).bbb, Int3(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).rrrr, Int4(1)))
        self.assertTrue(np.array_equal(Int3(1,2,3).gggg, Int4(2)))
        self.assertTrue(np.array_equal(Int3(1,2,3).bbbb, Int4(3)))
        self.assertTrue(np.array_equal(Int3(1,2,3).zyx, Int3(1,2,3).bgr))

        self.assertEqual(Int4(1,2,3,4).r, 1)
        self.assertEqual(Int4(1,2,3,4).g, 2)
        self.assertEqual(Int4(1,2,3,4).b, 3)
        self.assertEqual(Int4(1,2,3,4).a, 4)
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rgba, Int4(1,2,3,4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rgbr, Int4(1,2,3,1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).abgr, Int4(4,3,2,1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rrrr, Int4(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).gggg, Int4(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).bbbb, Int4(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).aaaa, Int4(4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rg, Int2(1,2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rr, Int2(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).gg, Int2(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).bb, Int2(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).aa, Int2(4)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).rrr, Int3(1)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).ggg, Int3(2)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).bbb, Int3(3)))
        self.assertTrue(np.array_equal(Int4(1,2,3,4).aaa, Int3(4)))

    def test_saturate(self):
        self.assertTrue(np.array_equal(saturate(Float2(-10,10)), Float2(0,1)))
        self.assertTrue(np.array_equal(saturate(Float3(-10,10,-10)), Float3(0,1,0)))
        self.assertTrue(np.array_equal(saturate(Float4(-10,10,-10,10)), Float4(0,1,0,1)))
        self.assertTrue(np.array_equal(saturate(Int2(-10,10)), Int2(0,1)))
        self.assertTrue(np.array_equal(saturate(Int3(-10,10,-10)), Int3(0,1,0)))
        self.assertTrue(np.array_equal(saturate(Int4(-10,10,-10,10)), Int4(0,1,0,1)))

    def test_sign(self):
        self.assertTrue(np.array_equal(sign(Float2(-10,10)), Float2(-1,1)))
        self.assertTrue(np.array_equal(sign(Float3(-10,10,-10)), Float3(-1,1,-1)))
        self.assertTrue(np.array_equal(sign(Float4(-10,10,-10,10)), Float4(-1,1,-1,1)))
        self.assertTrue(np.array_equal(sign(Int2(-10,10)), Int2(-1,1)))
        self.assertTrue(np.array_equal(sign(Int3(-10,10,-10)), Int3(-1,1,-1)))
        self.assertTrue(np.array_equal(sign(Int4(-10,10,-10,10)), Int4(-1,1,-1,1)))

        self.assertTrue(np.array_equal(sign(Float2(0)), Float2(0)))
        self.assertTrue(np.array_equal(sign(Float3(0)), Float3(0)))
        self.assertTrue(np.array_equal(sign(Float4(0)), Float4(0)))
        self.assertTrue(np.array_equal(sign(Int2(0)), Int2(0)))
        self.assertTrue(np.array_equal(sign(Int3(0)), Int3(0)))
        self.assertTrue(np.array_equal(sign(Int4(0)), Int4(0)))

    def test_lerp(self):
        self.assertTrue(np.array_equal(lerp(Float2(0),Float2(10),0.5), Float2(5)))
        self.assertTrue(np.array_equal(lerp(Float3(0),Float3(10),0.5), Float3(5)))
        self.assertTrue(np.array_equal(lerp(Float4(0),Float4(10),0.5), Float4(5)))

    def test_one(self):
        self.assertTrue(np.array_equal(Float2(1), Float2.one()))
        self.assertTrue(np.array_equal(Float3(1), Float3.one()))
        self.assertTrue(np.array_equal(Float4(1), Float4.one()))

    def test_zero(self):
        self.assertTrue(np.array_equal(Float2(0), Float2.zero()))
        self.assertTrue(np.array_equal(Float3(0), Float3.zero()))
        self.assertTrue(np.array_equal(Float4(0), Float4.zero()))

    def test_x_axis(self):
        self.assertTrue(np.array_equal(Float2.x_axis(), Float2(1,0)))
        self.assertTrue(np.array_equal(Float3.x_axis(), Float3(1,0,0)))
        self.assertTrue(np.array_equal(Float4.x_axis(), Float4(1,0,0,0)))

    def test_y_axis(self):
        self.assertTrue(np.array_equal(Float2.y_axis(), Float2(0,1)))
        self.assertTrue(np.array_equal(Float3.y_axis(), Float3(0,1,0)))
        self.assertTrue(np.array_equal(Float4.y_axis(), Float4(0,1,0,0)))

    def test_z_axis(self):
        self.assertTrue(np.array_equal(Float3.z_axis(), Float3(0,0,1)))
        self.assertTrue(np.array_equal(Float4.z_axis(), Float4(0,0,1,0)))

if __name__ == '__main__':
    unittest.main()