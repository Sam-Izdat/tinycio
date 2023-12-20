import torch
import numpy as np

# Python-land syntactic aspartame
# NOTE: All color-specific code should stay out of this base module

class Float2(np.ndarray):
    """
    Float2 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 2, "list/tuple must have 2 components"
                arr = np.asarray([args[0][0], args[0][1]], dtype=np.float32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 2, \
                    "numpy array must be sized [C=2] or [C=2, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.float32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 2, \
                    "torch tensor must be sized [C=2] or [C=2, H=1, W=1]"
                value = args[0].squeeze().float().cpu()
                arr = np.asarray([value[0].item(), value[1].item()], dtype=np.float32).view(cls)
            else:
                value = float(args[0])
                arr = np.asarray([value, value], dtype=np.float32).view(cls)
        elif len(args) == 2:
            arr = np.asarray(args, dtype=np.float32).view(cls)
        else: 
            raise TypeError("Float2 only accepts 1 or 2 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1])

    @property
    def x(self) -> float:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> float:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def r(self) -> float:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> float:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value

    @staticmethod
    def zero():
        """Returns numeric type filled with zero values"""
        return Float2(0., 0.)
    @staticmethod
    def one():
        """Returns numeric type filled with one values"""
        return Float2(1., 1.)
    @staticmethod
    def x_axis():
        """Returns numeric type with x-axis set to 1 and all others to 0"""
        return Float2(1., 0.)
    @staticmethod
    def y_axis():
        """Returns numeric type with y-axis set to 1 and all others to 0"""
        return Float2(0., 1.)

    @property
    def xx(self): return Float2(self.x, self.x)
    @property
    def xy(self): return self
    @property
    def yx(self): return Float2(self.y, self.x)
    @property
    def yy(self): return Float2(self.y, self.y)

    @property
    def rr(self): return Float2(self.r, self.r)
    @property
    def rg(self): return self
    @property
    def gr(self): return Float2(self.g, self.r)
    @property
    def gg(self): return Float2(self.g, self.g)

    @property
    def xxx(self): return Float3(self.x, self.x, self.x)
    @property
    def xxy(self): return Float3(self.x, self.x, self.y)
    @property
    def xyx(self): return Float3(self.x, self.y, self.x)
    @property
    def xyy(self): return Float3(self.x, self.y, self.y)
    @property
    def yxx(self): return Float3(self.y, self.x, self.x)
    @property
    def yxy(self): return Float3(self.y, self.x, self.y)
    @property
    def yyx(self): return Float3(self.y, self.y, self.x)
    @property
    def yyy(self): return Float3(self.y, self.y, self.y)

    @property
    def rrr(self): return Float3(self.r, self.r, self.r)
    @property
    def rrg(self): return Float3(self.r, self.r, self.g)
    @property
    def rgr(self): return Float3(self.r, self.g, self.r)
    @property
    def rgg(self): return Float3(self.r, self.g, self.g)
    @property
    def grr(self): return Float3(self.g, self.r, self.r)
    @property
    def grg(self): return Float3(self.g, self.r, self.g)
    @property
    def ggr(self): return Float3(self.g, self.g, self.r)
    @property
    def ggg(self): return Float3(self.g, self.g, self.g)

    @property
    def xxxx(self): return Float4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Float4(self.x, self.x, self.x, self.y)
    @property
    def xxyx(self): return Float4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Float4(self.x, self.x, self.y, self.y)
    @property
    def xyxx(self): return Float4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Float4(self.x, self.y, self.x, self.y)
    @property
    def xyyx(self): return Float4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Float4(self.x, self.y, self.y, self.y)
    @property
    def yxxx(self): return Float4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Float4(self.y, self.x, self.x, self.y)
    @property
    def yxyx(self): return Float4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Float4(self.y, self.x, self.y, self.y)
    @property
    def yyxx(self): return Float4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Float4(self.y, self.y, self.x, self.y)
    @property
    def yyyx(self): return Float4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Float4(self.y, self.y, self.y, self.y)

    @property
    def rrrr(self): return Float4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Float4(self.r, self.r, self.r, self.g)
    @property
    def rrgr(self): return Float4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Float4(self.r, self.r, self.g, self.g)
    @property
    def rgrr(self): return Float4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Float4(self.r, self.g, self.r, self.g)
    @property
    def rggr(self): return Float4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Float4(self.r, self.g, self.g, self.g)
    @property
    def grrr(self): return Float4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Float4(self.g, self.r, self.r, self.g)
    @property
    def grgr(self): return Float4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Float4(self.g, self.r, self.g, self.g)
    @property
    def ggrr(self): return Float4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Float4(self.g, self.g, self.r, self.g)
    @property
    def gggr(self): return Float4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Float4(self.g, self.g, self.g, self.g)

class Float3(np.ndarray):
    """
    Float3 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 3, "list/tuple must have 3 components"
                arr = np.asarray([args[0][0], args[0][1], args[0][2]], dtype=np.float32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 3, \
                    "numpy array must be sized [C=3] or [C=3, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.float32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 3, \
                    "torch tensor must be sized [C=3] or [C=3, H=1, W=1]"
                value = args[0].squeeze().float().cpu()
                arr = np.asarray([value[0].item(), value[1].item(), value[2].item()], dtype=np.float32).view(cls)
            else:
                value = float(args[0])
                arr = np.asarray([value, value, value], dtype=np.float32).view(cls)
        elif len(args) == 3:
            arr = np.asarray(args, dtype=np.float32).view(cls)
        else: 
            raise TypeError("Float3 only accepts 1 or 3 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1], self[2]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1], self[2])

    @property
    def x(self) -> float:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> float:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def z(self) -> float:
        return self[2]
    @z.setter
    def z(self, value):
        self[2] = value
    @property
    def r(self) -> float:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> float:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value
    @property
    def b(self) -> float:
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value
    @staticmethod
    def zero():
        """Returns numeric type filled with zero values"""
        return Float3(0., 0., 0.)
    @staticmethod
    def one():
        """Returns numeric type filled with one values"""
        return Float3(1., 1., 1.)
    @staticmethod
    def x_axis():
        """Returns numeric type with x-axis set to 1 and all others to 0"""
        return Float3(1., 0., 0.)
    @staticmethod
    def y_axis():
        """Returns numeric type with y-axis set to 1 and all others to 0"""
        return Float3(0., 1., 0.)
    @staticmethod
    def z_axis():
        """Returns numeric type with z-axis set to 1 and all others to 0"""
        return Float3(0., 0., 1.)

    @property
    def xx(self): return Float2(self.x, self.x)
    @property
    def xy(self): return Float2(self.x, self.y)
    @property
    def xz(self): return Float2(self.x, self.z)
    @property
    def yx(self): return Float2(self.y, self.x)
    @property
    def yy(self): return Float2(self.y, self.y)
    @property
    def yz(self): return Float2(self.y, self.z)
    @property
    def zx(self): return Float2(self.z, self.x)
    @property
    def zy(self): return Float2(self.z, self.y)
    @property
    def zz(self): return Float2(self.z, self.z)

    @property
    def rr(self): return Float2(self.r, self.r)
    @property
    def rg(self): return Float2(self.r, self.g)
    @property
    def rb(self): return Float2(self.r, self.b)
    @property
    def gr(self): return Float2(self.g, self.r)
    @property
    def gg(self): return Float2(self.g, self.g)
    @property
    def gb(self): return Float2(self.g, self.b)
    @property
    def br(self): return Float2(self.b, self.r)
    @property
    def bg(self): return Float2(self.b, self.g)
    @property
    def bb(self): return Float2(self.b, self.b)

    @property
    def xxx(self): return Float3(self.x, self.x, self.x)
    @property
    def xxy(self): return Float3(self.x, self.x, self.y)
    @property
    def xxz(self): return Float3(self.x, self.x, self.z)
    @property
    def xyx(self): return Float3(self.x, self.y, self.x)
    @property
    def xyy(self): return Float3(self.x, self.y, self.y)
    @property
    def xyz(self): return self
    @property
    def xzx(self): return Float3(self.x, self.z, self.x)
    @property
    def xzy(self): return Float3(self.x, self.z, self.y)
    @property
    def xzz(self): return Float3(self.x, self.z, self.z)
    @property
    def yxx(self): return Float3(self.y, self.x, self.x)
    @property
    def yxy(self): return Float3(self.y, self.x, self.y)
    @property
    def yxz(self): return Float3(self.y, self.x, self.z)
    @property
    def yyx(self): return Float3(self.y, self.y, self.x)
    @property
    def yyy(self): return Float3(self.y, self.y, self.y)
    @property
    def yyz(self): return Float3(self.y, self.y, self.z)
    @property
    def yzx(self): return Float3(self.y, self.z, self.x)
    @property
    def yzy(self): return Float3(self.y, self.z, self.y)
    @property
    def yzz(self): return Float3(self.y, self.z, self.z)
    @property
    def zxx(self): return Float3(self.z, self.x, self.x)
    @property
    def zxy(self): return Float3(self.z, self.x, self.y)
    @property
    def zxz(self): return Float3(self.z, self.x, self.z)
    @property
    def zyx(self): return Float3(self.z, self.y, self.x)
    @property
    def zyy(self): return Float3(self.z, self.y, self.y)
    @property
    def zyz(self): return Float3(self.z, self.y, self.z)
    @property
    def zzx(self): return Float3(self.z, self.z, self.x)
    @property
    def zzy(self): return Float3(self.z, self.z, self.y)
    @property
    def zzz(self): return Float3(self.z, self.z, self.z)

    @property
    def rrr(self): return Float3(self.r, self.r, self.r)
    @property
    def rrg(self): return Float3(self.r, self.r, self.g)
    @property
    def rrb(self): return Float3(self.r, self.r, self.b)
    @property
    def rgr(self): return Float3(self.r, self.g, self.r)
    @property
    def rgg(self): return Float3(self.r, self.g, self.g)
    @property
    def rgb(self): return self
    @property
    def rbr(self): return Float3(self.r, self.b, self.r)
    @property
    def rbg(self): return Float3(self.r, self.b, self.g)
    @property
    def rbb(self): return Float3(self.r, self.b, self.b)
    @property
    def grr(self): return Float3(self.g, self.r, self.r)
    @property
    def grg(self): return Float3(self.g, self.r, self.g)
    @property
    def grb(self): return Float3(self.g, self.r, self.b)
    @property
    def ggr(self): return Float3(self.g, self.g, self.r)
    @property
    def ggg(self): return Float3(self.g, self.g, self.g)
    @property
    def ggb(self): return Float3(self.g, self.g, self.b)
    @property
    def gbr(self): return Float3(self.g, self.b, self.r)
    @property
    def gbg(self): return Float3(self.g, self.b, self.g)
    @property
    def gbb(self): return Float3(self.g, self.b, self.b)
    @property
    def brr(self): return Float3(self.b, self.r, self.r)
    @property
    def brg(self): return Float3(self.b, self.r, self.g)
    @property
    def brb(self): return Float3(self.b, self.r, self.b)
    @property
    def bgr(self): return Float3(self.b, self.g, self.r)
    @property
    def bgg(self): return Float3(self.b, self.g, self.g)
    @property
    def bgb(self): return Float3(self.b, self.g, self.b)
    @property
    def bbr(self): return Float3(self.b, self.b, self.r)
    @property
    def bbg(self): return Float3(self.b, self.b, self.g)
    @property
    def bbb(self): return Float3(self.b, self.b, self.b)

    @property
    def xxxx(self): return Float4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Float4(self.x, self.x, self.x, self.y)
    @property
    def xxxz(self): return Float4(self.x, self.x, self.x, self.z)
    @property
    def xxyx(self): return Float4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Float4(self.x, self.x, self.y, self.y)
    @property
    def xxyz(self): return Float4(self.x, self.x, self.y, self.z)
    @property
    def xxzx(self): return Float4(self.x, self.x, self.z, self.x)
    @property
    def xxzy(self): return Float4(self.x, self.x, self.z, self.y)
    @property
    def xxzz(self): return Float4(self.x, self.x, self.z, self.z)
    @property
    def xyxx(self): return Float4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Float4(self.x, self.y, self.x, self.y)
    @property
    def xyxz(self): return Float4(self.x, self.y, self.x, self.z)
    @property
    def xyyx(self): return Float4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Float4(self.x, self.y, self.y, self.y)
    @property
    def xyyz(self): return Float4(self.x, self.y, self.y, self.z)
    @property
    def xyzx(self): return Float4(self.x, self.y, self.z, self.x)
    @property
    def xyzy(self): return Float4(self.x, self.y, self.z, self.y)
    @property
    def xyzz(self): return Float4(self.x, self.y, self.z, self.z)
    @property
    def xzxx(self): return Float4(self.x, self.z, self.x, self.x)
    @property
    def xzxy(self): return Float4(self.x, self.z, self.x, self.y)
    @property
    def xzxz(self): return Float4(self.x, self.z, self.x, self.z)
    @property
    def xzyx(self): return Float4(self.x, self.z, self.y, self.x)
    @property
    def xzyy(self): return Float4(self.x, self.z, self.y, self.y)
    @property
    def xzyz(self): return Float4(self.x, self.z, self.y, self.z)
    @property
    def xzzx(self): return Float4(self.x, self.z, self.z, self.x)
    @property
    def xzzy(self): return Float4(self.x, self.z, self.z, self.y)
    @property
    def xzzz(self): return Float4(self.x, self.z, self.z, self.z)
    @property
    def yxxx(self): return Float4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Float4(self.y, self.x, self.x, self.y)
    @property
    def yxxz(self): return Float4(self.y, self.x, self.x, self.z)
    @property
    def yxyx(self): return Float4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Float4(self.y, self.x, self.y, self.y)
    @property
    def yxyz(self): return Float4(self.y, self.x, self.y, self.z)
    @property
    def yxzx(self): return Float4(self.y, self.x, self.z, self.x)
    @property
    def yxzy(self): return Float4(self.y, self.x, self.z, self.y)
    @property
    def yxzz(self): return Float4(self.y, self.x, self.z, self.z)
    @property
    def yyxx(self): return Float4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Float4(self.y, self.y, self.x, self.y)
    @property
    def yyxz(self): return Float4(self.y, self.y, self.x, self.z)
    @property
    def yyyx(self): return Float4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Float4(self.y, self.y, self.y, self.y)
    @property
    def yyyz(self): return Float4(self.y, self.y, self.y, self.z)
    @property
    def yyzx(self): return Float4(self.y, self.y, self.z, self.x)
    @property
    def yyzy(self): return Float4(self.y, self.y, self.z, self.y)
    @property
    def yyzz(self): return Float4(self.y, self.y, self.z, self.z)
    @property
    def yzxx(self): return Float4(self.y, self.z, self.x, self.x)
    @property
    def yzxy(self): return Float4(self.y, self.z, self.x, self.y)
    @property
    def yzxz(self): return Float4(self.y, self.z, self.x, self.z)
    @property
    def yzyx(self): return Float4(self.y, self.z, self.y, self.x)
    @property
    def yzyy(self): return Float4(self.y, self.z, self.y, self.y)
    @property
    def yzyz(self): return Float4(self.y, self.z, self.y, self.z)
    @property
    def yzzx(self): return Float4(self.y, self.z, self.z, self.x)
    @property
    def yzzy(self): return Float4(self.y, self.z, self.z, self.y)
    @property
    def yzzz(self): return Float4(self.y, self.z, self.z, self.z)
    @property
    def zxxx(self): return Float4(self.z, self.x, self.x, self.x)
    @property
    def zxxy(self): return Float4(self.z, self.x, self.x, self.y)
    @property
    def zxxz(self): return Float4(self.z, self.x, self.x, self.z)
    @property
    def zxyx(self): return Float4(self.z, self.x, self.y, self.x)
    @property
    def zxyy(self): return Float4(self.z, self.x, self.y, self.y)
    @property
    def zxyz(self): return Float4(self.z, self.x, self.y, self.z)
    @property
    def zxzx(self): return Float4(self.z, self.x, self.z, self.x)
    @property
    def zxzy(self): return Float4(self.z, self.x, self.z, self.y)
    @property
    def zxzz(self): return Float4(self.z, self.x, self.z, self.z)
    @property
    def zyxx(self): return Float4(self.z, self.y, self.x, self.x)
    @property
    def zyxy(self): return Float4(self.z, self.y, self.x, self.y)
    @property
    def zyxz(self): return Float4(self.z, self.y, self.x, self.z)
    @property
    def zyyx(self): return Float4(self.z, self.y, self.y, self.x)
    @property
    def zyyy(self): return Float4(self.z, self.y, self.y, self.y)
    @property
    def zyyz(self): return Float4(self.z, self.y, self.y, self.z)
    @property
    def zyzx(self): return Float4(self.z, self.y, self.z, self.x)
    @property
    def zyzy(self): return Float4(self.z, self.y, self.z, self.y)
    @property
    def zyzz(self): return Float4(self.z, self.y, self.z, self.z)
    @property
    def zzxx(self): return Float4(self.z, self.z, self.x, self.x)
    @property
    def zzxy(self): return Float4(self.z, self.z, self.x, self.y)
    @property
    def zzxz(self): return Float4(self.z, self.z, self.x, self.z)
    @property
    def zzyx(self): return Float4(self.z, self.z, self.y, self.x)
    @property
    def zzyy(self): return Float4(self.z, self.z, self.y, self.y)
    @property
    def zzyz(self): return Float4(self.z, self.z, self.y, self.z)
    @property
    def zzzx(self): return Float4(self.z, self.z, self.z, self.x)
    @property
    def zzzy(self): return Float4(self.z, self.z, self.z, self.y)
    @property
    def zzzz(self): return Float4(self.z, self.z, self.z, self.z)

    @property
    def rrrr(self): return Float4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Float4(self.r, self.r, self.r, self.g)
    @property
    def rrrb(self): return Float4(self.r, self.r, self.r, self.b)
    @property
    def rrgr(self): return Float4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Float4(self.r, self.r, self.g, self.g)
    @property
    def rrgb(self): return Float4(self.r, self.r, self.g, self.b)
    @property
    def rrbr(self): return Float4(self.r, self.r, self.b, self.r)
    @property
    def rrbg(self): return Float4(self.r, self.r, self.b, self.g)
    @property
    def rrbb(self): return Float4(self.r, self.r, self.b, self.b)
    @property
    def rgrr(self): return Float4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Float4(self.r, self.g, self.r, self.g)
    @property
    def rgrb(self): return Float4(self.r, self.g, self.r, self.b)
    @property
    def rggr(self): return Float4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Float4(self.r, self.g, self.g, self.g)
    @property
    def rggb(self): return Float4(self.r, self.g, self.g, self.b)
    @property
    def rgbr(self): return Float4(self.r, self.g, self.b, self.r)
    @property
    def rgbg(self): return Float4(self.r, self.g, self.b, self.g)
    @property
    def rgbb(self): return Float4(self.r, self.g, self.b, self.b)
    @property
    def rbrr(self): return Float4(self.r, self.b, self.r, self.r)
    @property
    def rbrg(self): return Float4(self.r, self.b, self.r, self.g)
    @property
    def rbrb(self): return Float4(self.r, self.b, self.r, self.b)
    @property
    def rbgr(self): return Float4(self.r, self.b, self.g, self.r)
    @property
    def rbgg(self): return Float4(self.r, self.b, self.g, self.g)
    @property
    def rbgb(self): return Float4(self.r, self.b, self.g, self.b)
    @property
    def rbbr(self): return Float4(self.r, self.b, self.b, self.r)
    @property
    def rbbg(self): return Float4(self.r, self.b, self.b, self.g)
    @property
    def rbbb(self): return Float4(self.r, self.b, self.b, self.b)
    @property
    def grrr(self): return Float4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Float4(self.g, self.r, self.r, self.g)
    @property
    def grrb(self): return Float4(self.g, self.r, self.r, self.b)
    @property
    def grgr(self): return Float4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Float4(self.g, self.r, self.g, self.g)
    @property
    def grgb(self): return Float4(self.g, self.r, self.g, self.b)
    @property
    def grbr(self): return Float4(self.g, self.r, self.b, self.r)
    @property
    def grbg(self): return Float4(self.g, self.r, self.b, self.g)
    @property
    def grbb(self): return Float4(self.g, self.r, self.b, self.b)
    @property
    def ggrr(self): return Float4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Float4(self.g, self.g, self.r, self.g)
    @property
    def ggrb(self): return Float4(self.g, self.g, self.r, self.b)
    @property
    def gggr(self): return Float4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Float4(self.g, self.g, self.g, self.g)
    @property
    def gggb(self): return Float4(self.g, self.g, self.g, self.b)
    @property
    def ggbr(self): return Float4(self.g, self.g, self.b, self.r)
    @property
    def ggbg(self): return Float4(self.g, self.g, self.b, self.g)
    @property
    def ggbb(self): return Float4(self.g, self.g, self.b, self.b)
    @property
    def gbrr(self): return Float4(self.g, self.b, self.r, self.r)
    @property
    def gbrg(self): return Float4(self.g, self.b, self.r, self.g)
    @property
    def gbrb(self): return Float4(self.g, self.b, self.r, self.b)
    @property
    def gbgr(self): return Float4(self.g, self.b, self.g, self.r)
    @property
    def gbgg(self): return Float4(self.g, self.b, self.g, self.g)
    @property
    def gbgb(self): return Float4(self.g, self.b, self.g, self.b)
    @property
    def gbbr(self): return Float4(self.g, self.b, self.b, self.r)
    @property
    def gbbg(self): return Float4(self.g, self.b, self.b, self.g)
    @property
    def gbbb(self): return Float4(self.g, self.b, self.b, self.b)
    @property
    def brrr(self): return Float4(self.b, self.r, self.r, self.r)
    @property
    def brrg(self): return Float4(self.b, self.r, self.r, self.g)
    @property
    def brrb(self): return Float4(self.b, self.r, self.r, self.b)
    @property
    def brgr(self): return Float4(self.b, self.r, self.g, self.r)
    @property
    def brgg(self): return Float4(self.b, self.r, self.g, self.g)
    @property
    def brgb(self): return Float4(self.b, self.r, self.g, self.b)
    @property
    def brbr(self): return Float4(self.b, self.r, self.b, self.r)
    @property
    def brbg(self): return Float4(self.b, self.r, self.b, self.g)
    @property
    def brbb(self): return Float4(self.b, self.r, self.b, self.b)
    @property
    def bgrr(self): return Float4(self.b, self.g, self.r, self.r)
    @property
    def bgrg(self): return Float4(self.b, self.g, self.r, self.g)
    @property
    def bgrb(self): return Float4(self.b, self.g, self.r, self.b)
    @property
    def bggr(self): return Float4(self.b, self.g, self.g, self.r)
    @property
    def bggg(self): return Float4(self.b, self.g, self.g, self.g)
    @property
    def bggb(self): return Float4(self.b, self.g, self.g, self.b)
    @property
    def bgbr(self): return Float4(self.b, self.g, self.b, self.r)
    @property
    def bgbg(self): return Float4(self.b, self.g, self.b, self.g)
    @property
    def bgbb(self): return Float4(self.b, self.g, self.b, self.b)
    @property
    def bbrr(self): return Float4(self.b, self.b, self.r, self.r)
    @property
    def bbrg(self): return Float4(self.b, self.b, self.r, self.g)
    @property
    def bbrb(self): return Float4(self.b, self.b, self.r, self.b)
    @property
    def bbgr(self): return Float4(self.b, self.b, self.g, self.r)
    @property
    def bbgg(self): return Float4(self.b, self.b, self.g, self.g)
    @property
    def bbgb(self): return Float4(self.b, self.b, self.g, self.b)
    @property
    def bbbr(self): return Float4(self.b, self.b, self.b, self.r)
    @property
    def bbbg(self): return Float4(self.b, self.b, self.b, self.g)
    @property
    def bbbb(self): return Float4(self.b, self.b, self.b, self.b)

class Float4(np.ndarray):
    """
    Float4 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 4, "list/tuple must have 4 components"
                arr = np.asarray([args[0][0], args[0][1], args[0][2], args[0][3]], dtype=np.float32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 4, \
                    "numpy array must be sized [C=4] or [C=4, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.float32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 4, \
                    "torch tensor must be sized [C=4] or [C=4, H=1, W=1]"
                value = args[0].squeeze().float().cpu()
                arr = np.asarray([value[0].item(), value[1].item(), value[2].item(), value[3].item()], dtype=np.float32).view(cls)
            else:
                value = float(args[0])
                arr = np.asarray([value, value, value, value], dtype=np.float32).view(cls)
        elif len(args) == 4:
            arr = np.asarray(args, dtype=np.float32).view(cls)
        else: 
            raise TypeError("Float4 only accepts 1 or 4 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1], self[2], self[3]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1], self[2], self[3])

    @property
    def x(self) -> float:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> float:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def z(self) -> float:
        return self[2]
    @z.setter
    def z(self, value):
        self[2] = value
    @property
    def w(self) -> float:
        return self[3]
    @w.setter
    def w(self, value):
        self[3] = value
    @property
    def r(self) -> float:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> float:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value
    @property
    def b(self) -> float:
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value
    @property
    def a(self) -> float:
        return self[3]
    @a.setter
    def a(self, value):
        self[3] = value
    @staticmethod
    def zero():
        """Returns numeric type filled with zero values"""
        return Float4(0., 0., 0., 0.)
    @staticmethod
    def one():
        """Returns numeric type filled with one values"""
        return Float4(1., 1., 1., 1.)
    @staticmethod
    def x_axis():
        """Returns numeric type with x-axis set to 1 and all others to 0"""
        return Float4(1., 0., 0., 0.)
    @staticmethod
    def y_axis():
        """Returns numeric type with y-axis set to 1 and all others to 0"""
        return Float4(0., 1., 0., 0.)
    @staticmethod
    def z_axis():
        """Returns numeric type with z-axis set to 1 and all others to 0"""
        return Float4(0., 0., 1., 0.)

    @property
    def xx(self): return Float2(self.x, self.x)
    @property
    def xy(self): return Float2(self.x, self.y)
    @property
    def xz(self): return Float2(self.x, self.z)
    @property
    def xw(self): return Float2(self.x, self.w)
    @property
    def yx(self): return Float2(self.y, self.x)
    @property
    def yy(self): return Float2(self.y, self.y)
    @property
    def yz(self): return Float2(self.y, self.z)
    @property
    def yw(self): return Float2(self.y, self.w)
    @property
    def zx(self): return Float2(self.z, self.x)
    @property
    def zy(self): return Float2(self.z, self.y)
    @property
    def zz(self): return Float2(self.z, self.z)
    @property
    def zw(self): return Float2(self.z, self.w)
    @property
    def wx(self): return Float2(self.w, self.x)
    @property
    def wy(self): return Float2(self.w, self.y)
    @property
    def wz(self): return Float2(self.w, self.z)
    @property
    def ww(self): return Float2(self.w, self.w)

    @property
    def rr(self): return Float2(self.r, self.r)
    @property
    def rg(self): return Float2(self.r, self.g)
    @property
    def rb(self): return Float2(self.r, self.b)
    @property
    def ra(self): return Float2(self.r, self.a)
    @property
    def gr(self): return Float2(self.g, self.r)
    @property
    def gg(self): return Float2(self.g, self.g)
    @property
    def gb(self): return Float2(self.g, self.b)
    @property
    def ga(self): return Float2(self.g, self.a)
    @property
    def br(self): return Float2(self.b, self.r)
    @property
    def bg(self): return Float2(self.b, self.g)
    @property
    def bb(self): return Float2(self.b, self.b)
    @property
    def ba(self): return Float2(self.b, self.a)
    @property
    def ar(self): return Float2(self.a, self.r)
    @property
    def ag(self): return Float2(self.a, self.g)
    @property
    def ab(self): return Float2(self.a, self.b)
    @property
    def aa(self): return Float2(self.a, self.a)

    @property
    def rrr(self): return Float3(self.r, self.r, self.r)
    @property
    def rrg(self): return Float3(self.r, self.r, self.g)
    @property
    def rrb(self): return Float3(self.r, self.r, self.b)
    @property
    def rra(self): return Float3(self.r, self.r, self.a)
    @property
    def rgr(self): return Float3(self.r, self.g, self.r)
    @property
    def rgg(self): return Float3(self.r, self.g, self.g)
    @property
    def rgb(self): return Float3(self.r, self.g, self.b)
    @property
    def rga(self): return Float3(self.r, self.g, self.a)
    @property
    def rbr(self): return Float3(self.r, self.b, self.r)
    @property
    def rbg(self): return Float3(self.r, self.b, self.g)
    @property
    def rbb(self): return Float3(self.r, self.b, self.b)
    @property
    def rba(self): return Float3(self.r, self.b, self.a)
    @property
    def rar(self): return Float3(self.r, self.a, self.r)
    @property
    def rag(self): return Float3(self.r, self.a, self.g)
    @property
    def rab(self): return Float3(self.r, self.a, self.b)
    @property
    def raa(self): return Float3(self.r, self.a, self.a)
    @property
    def grr(self): return Float3(self.g, self.r, self.r)
    @property
    def grg(self): return Float3(self.g, self.r, self.g)
    @property
    def grb(self): return Float3(self.g, self.r, self.b)
    @property
    def gra(self): return Float3(self.g, self.r, self.a)
    @property
    def ggr(self): return Float3(self.g, self.g, self.r)
    @property
    def ggg(self): return Float3(self.g, self.g, self.g)
    @property
    def ggb(self): return Float3(self.g, self.g, self.b)
    @property
    def gga(self): return Float3(self.g, self.g, self.a)
    @property
    def gbr(self): return Float3(self.g, self.b, self.r)
    @property
    def gbg(self): return Float3(self.g, self.b, self.g)
    @property
    def gbb(self): return Float3(self.g, self.b, self.b)
    @property
    def gba(self): return Float3(self.g, self.b, self.a)
    @property
    def gar(self): return Float3(self.g, self.a, self.r)
    @property
    def gag(self): return Float3(self.g, self.a, self.g)
    @property
    def gab(self): return Float3(self.g, self.a, self.b)
    @property
    def gaa(self): return Float3(self.g, self.a, self.a)
    @property
    def brr(self): return Float3(self.b, self.r, self.r)
    @property
    def brg(self): return Float3(self.b, self.r, self.g)
    @property
    def brb(self): return Float3(self.b, self.r, self.b)
    @property
    def bra(self): return Float3(self.b, self.r, self.a)
    @property
    def bgr(self): return Float3(self.b, self.g, self.r)
    @property
    def bgg(self): return Float3(self.b, self.g, self.g)
    @property
    def bgb(self): return Float3(self.b, self.g, self.b)
    @property
    def bga(self): return Float3(self.b, self.g, self.a)
    @property
    def bbr(self): return Float3(self.b, self.b, self.r)
    @property
    def bbg(self): return Float3(self.b, self.b, self.g)
    @property
    def bbb(self): return Float3(self.b, self.b, self.b)
    @property
    def bba(self): return Float3(self.b, self.b, self.a)
    @property
    def bar(self): return Float3(self.b, self.a, self.r)
    @property
    def bag(self): return Float3(self.b, self.a, self.g)
    @property
    def bab(self): return Float3(self.b, self.a, self.b)
    @property
    def baa(self): return Float3(self.b, self.a, self.a)
    @property
    def arr(self): return Float3(self.a, self.r, self.r)
    @property
    def arg(self): return Float3(self.a, self.r, self.g)
    @property
    def arb(self): return Float3(self.a, self.r, self.b)
    @property
    def ara(self): return Float3(self.a, self.r, self.a)
    @property
    def agr(self): return Float3(self.a, self.g, self.r)
    @property
    def agg(self): return Float3(self.a, self.g, self.g)
    @property
    def agb(self): return Float3(self.a, self.g, self.b)
    @property
    def aga(self): return Float3(self.a, self.g, self.a)
    @property
    def abr(self): return Float3(self.a, self.b, self.r)
    @property
    def abg(self): return Float3(self.a, self.b, self.g)
    @property
    def abb(self): return Float3(self.a, self.b, self.b)
    @property
    def aba(self): return Float3(self.a, self.b, self.a)
    @property
    def aar(self): return Float3(self.a, self.a, self.r)
    @property
    def aag(self): return Float3(self.a, self.a, self.g)
    @property
    def aab(self): return Float3(self.a, self.a, self.b)
    @property
    def aaa(self): return Float3(self.a, self.a, self.a)

    @property
    def xxx(self): return Float3(self.x, self.x, self.x)
    @property
    def xxy(self): return Float3(self.x, self.x, self.y)
    @property
    def xxz(self): return Float3(self.x, self.x, self.z)
    @property
    def xxw(self): return Float3(self.x, self.x, self.w)
    @property
    def xyx(self): return Float3(self.x, self.y, self.x)
    @property
    def xyy(self): return Float3(self.x, self.y, self.y)
    @property
    def xyz(self): return Float3(self.x, self.y, self.z)
    @property
    def xyw(self): return Float3(self.x, self.y, self.w)
    @property
    def xzx(self): return Float3(self.x, self.z, self.x)
    @property
    def xzy(self): return Float3(self.x, self.z, self.y)
    @property
    def xzz(self): return Float3(self.x, self.z, self.z)
    @property
    def xzw(self): return Float3(self.x, self.z, self.w)
    @property
    def xwx(self): return Float3(self.x, self.w, self.x)
    @property
    def xwy(self): return Float3(self.x, self.w, self.y)
    @property
    def xwz(self): return Float3(self.x, self.w, self.z)
    @property
    def xww(self): return Float3(self.x, self.w, self.w)
    @property
    def yxx(self): return Float3(self.y, self.x, self.x)
    @property
    def yxy(self): return Float3(self.y, self.x, self.y)
    @property
    def yxz(self): return Float3(self.y, self.x, self.z)
    @property
    def yxw(self): return Float3(self.y, self.x, self.w)
    @property
    def yyx(self): return Float3(self.y, self.y, self.x)
    @property
    def yyy(self): return Float3(self.y, self.y, self.y)
    @property
    def yyz(self): return Float3(self.y, self.y, self.z)
    @property
    def yyw(self): return Float3(self.y, self.y, self.w)
    @property
    def yzx(self): return Float3(self.y, self.z, self.x)
    @property
    def yzy(self): return Float3(self.y, self.z, self.y)
    @property
    def yzz(self): return Float3(self.y, self.z, self.z)
    @property
    def yzw(self): return Float3(self.y, self.z, self.w)
    @property
    def ywx(self): return Float3(self.y, self.w, self.x)
    @property
    def ywy(self): return Float3(self.y, self.w, self.y)
    @property
    def ywz(self): return Float3(self.y, self.w, self.z)
    @property
    def yww(self): return Float3(self.y, self.w, self.w)
    @property
    def zxx(self): return Float3(self.z, self.x, self.x)
    @property
    def zxy(self): return Float3(self.z, self.x, self.y)
    @property
    def zxz(self): return Float3(self.z, self.x, self.z)
    @property
    def zxw(self): return Float3(self.z, self.x, self.w)
    @property
    def zyx(self): return Float3(self.z, self.y, self.x)
    @property
    def zyy(self): return Float3(self.z, self.y, self.y)
    @property
    def zyz(self): return Float3(self.z, self.y, self.z)
    @property
    def zyw(self): return Float3(self.z, self.y, self.w)
    @property
    def zzx(self): return Float3(self.z, self.z, self.x)
    @property
    def zzy(self): return Float3(self.z, self.z, self.y)
    @property
    def zzz(self): return Float3(self.z, self.z, self.z)
    @property
    def zzw(self): return Float3(self.z, self.z, self.w)
    @property
    def zwx(self): return Float3(self.z, self.w, self.x)
    @property
    def zwy(self): return Float3(self.z, self.w, self.y)
    @property
    def zwz(self): return Float3(self.z, self.w, self.z)
    @property
    def zww(self): return Float3(self.z, self.w, self.w)
    @property
    def wxx(self): return Float3(self.w, self.x, self.x)
    @property
    def wxy(self): return Float3(self.w, self.x, self.y)
    @property
    def wxz(self): return Float3(self.w, self.x, self.z)
    @property
    def wxw(self): return Float3(self.w, self.x, self.w)
    @property
    def wyx(self): return Float3(self.w, self.y, self.x)
    @property
    def wyy(self): return Float3(self.w, self.y, self.y)
    @property
    def wyz(self): return Float3(self.w, self.y, self.z)
    @property
    def wyw(self): return Float3(self.w, self.y, self.w)
    @property
    def wzx(self): return Float3(self.w, self.z, self.x)
    @property
    def wzy(self): return Float3(self.w, self.z, self.y)
    @property
    def wzz(self): return Float3(self.w, self.z, self.z)
    @property
    def wzw(self): return Float3(self.w, self.z, self.w)
    @property
    def wwx(self): return Float3(self.w, self.w, self.x)
    @property
    def wwy(self): return Float3(self.w, self.w, self.y)
    @property
    def wwz(self): return Float3(self.w, self.w, self.z)
    @property
    def www(self): return Float3(self.w, self.w, self.w)

    @property
    def xxxx(self): return Float4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Float4(self.x, self.x, self.x, self.y)
    @property
    def xxxz(self): return Float4(self.x, self.x, self.x, self.z)
    @property
    def xxxw(self): return Float4(self.x, self.x, self.x, self.w)
    @property
    def xxyx(self): return Float4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Float4(self.x, self.x, self.y, self.y)
    @property
    def xxyz(self): return Float4(self.x, self.x, self.y, self.z)
    @property
    def xxyw(self): return Float4(self.x, self.x, self.y, self.w)
    @property
    def xxzx(self): return Float4(self.x, self.x, self.z, self.x)
    @property
    def xxzy(self): return Float4(self.x, self.x, self.z, self.y)
    @property
    def xxzz(self): return Float4(self.x, self.x, self.z, self.z)
    @property
    def xxzw(self): return Float4(self.x, self.x, self.z, self.w)
    @property
    def xxwx(self): return Float4(self.x, self.x, self.w, self.x)
    @property
    def xxwy(self): return Float4(self.x, self.x, self.w, self.y)
    @property
    def xxwz(self): return Float4(self.x, self.x, self.w, self.z)
    @property
    def xxww(self): return Float4(self.x, self.x, self.w, self.w)
    @property
    def xyxx(self): return Float4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Float4(self.x, self.y, self.x, self.y)
    @property
    def xyxz(self): return Float4(self.x, self.y, self.x, self.z)
    @property
    def xyxw(self): return Float4(self.x, self.y, self.x, self.w)
    @property
    def xyyx(self): return Float4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Float4(self.x, self.y, self.y, self.y)
    @property
    def xyyz(self): return Float4(self.x, self.y, self.y, self.z)
    @property
    def xyyw(self): return Float4(self.x, self.y, self.y, self.w)
    @property
    def xyzx(self): return Float4(self.x, self.y, self.z, self.x)
    @property
    def xyzy(self): return Float4(self.x, self.y, self.z, self.y)
    @property
    def xyzz(self): return Float4(self.x, self.y, self.z, self.z)
    @property
    def xyzw(self): return self
    @property
    def xywx(self): return Float4(self.x, self.y, self.w, self.x)
    @property
    def xywy(self): return Float4(self.x, self.y, self.w, self.y)
    @property
    def xywz(self): return Float4(self.x, self.y, self.w, self.z)
    @property
    def xyww(self): return Float4(self.x, self.y, self.w, self.w)
    @property
    def xzxx(self): return Float4(self.x, self.z, self.x, self.x)
    @property
    def xzxy(self): return Float4(self.x, self.z, self.x, self.y)
    @property
    def xzxz(self): return Float4(self.x, self.z, self.x, self.z)
    @property
    def xzxw(self): return Float4(self.x, self.z, self.x, self.w)
    @property
    def xzyx(self): return Float4(self.x, self.z, self.y, self.x)
    @property
    def xzyy(self): return Float4(self.x, self.z, self.y, self.y)
    @property
    def xzyz(self): return Float4(self.x, self.z, self.y, self.z)
    @property
    def xzyw(self): return Float4(self.x, self.z, self.y, self.w)
    @property
    def xzzx(self): return Float4(self.x, self.z, self.z, self.x)
    @property
    def xzzy(self): return Float4(self.x, self.z, self.z, self.y)
    @property
    def xzzz(self): return Float4(self.x, self.z, self.z, self.z)
    @property
    def xzzw(self): return Float4(self.x, self.z, self.z, self.w)
    @property
    def xzwx(self): return Float4(self.x, self.z, self.w, self.x)
    @property
    def xzwy(self): return Float4(self.x, self.z, self.w, self.y)
    @property
    def xzwz(self): return Float4(self.x, self.z, self.w, self.z)
    @property
    def xzww(self): return Float4(self.x, self.z, self.w, self.w)
    @property
    def xwxx(self): return Float4(self.x, self.w, self.x, self.x)
    @property
    def xwxy(self): return Float4(self.x, self.w, self.x, self.y)
    @property
    def xwxz(self): return Float4(self.x, self.w, self.x, self.z)
    @property
    def xwxw(self): return Float4(self.x, self.w, self.x, self.w)
    @property
    def xwyx(self): return Float4(self.x, self.w, self.y, self.x)
    @property
    def xwyy(self): return Float4(self.x, self.w, self.y, self.y)
    @property
    def xwyz(self): return Float4(self.x, self.w, self.y, self.z)
    @property
    def xwyw(self): return Float4(self.x, self.w, self.y, self.w)
    @property
    def xwzx(self): return Float4(self.x, self.w, self.z, self.x)
    @property
    def xwzy(self): return Float4(self.x, self.w, self.z, self.y)
    @property
    def xwzz(self): return Float4(self.x, self.w, self.z, self.z)
    @property
    def xwzw(self): return Float4(self.x, self.w, self.z, self.w)
    @property
    def xwwx(self): return Float4(self.x, self.w, self.w, self.x)
    @property
    def xwwy(self): return Float4(self.x, self.w, self.w, self.y)
    @property
    def xwwz(self): return Float4(self.x, self.w, self.w, self.z)
    @property
    def xwww(self): return Float4(self.x, self.w, self.w, self.w)
    @property
    def yxxx(self): return Float4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Float4(self.y, self.x, self.x, self.y)
    @property
    def yxxz(self): return Float4(self.y, self.x, self.x, self.z)
    @property
    def yxxw(self): return Float4(self.y, self.x, self.x, self.w)
    @property
    def yxyx(self): return Float4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Float4(self.y, self.x, self.y, self.y)
    @property
    def yxyz(self): return Float4(self.y, self.x, self.y, self.z)
    @property
    def yxyw(self): return Float4(self.y, self.x, self.y, self.w)
    @property
    def yxzx(self): return Float4(self.y, self.x, self.z, self.x)
    @property
    def yxzy(self): return Float4(self.y, self.x, self.z, self.y)
    @property
    def yxzz(self): return Float4(self.y, self.x, self.z, self.z)
    @property
    def yxzw(self): return Float4(self.y, self.x, self.z, self.w)
    @property
    def yxwx(self): return Float4(self.y, self.x, self.w, self.x)
    @property
    def yxwy(self): return Float4(self.y, self.x, self.w, self.y)
    @property
    def yxwz(self): return Float4(self.y, self.x, self.w, self.z)
    @property
    def yxww(self): return Float4(self.y, self.x, self.w, self.w)
    @property
    def yyxx(self): return Float4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Float4(self.y, self.y, self.x, self.y)
    @property
    def yyxz(self): return Float4(self.y, self.y, self.x, self.z)
    @property
    def yyxw(self): return Float4(self.y, self.y, self.x, self.w)
    @property
    def yyyx(self): return Float4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Float4(self.y, self.y, self.y, self.y)
    @property
    def yyyz(self): return Float4(self.y, self.y, self.y, self.z)
    @property
    def yyyw(self): return Float4(self.y, self.y, self.y, self.w)
    @property
    def yyzx(self): return Float4(self.y, self.y, self.z, self.x)
    @property
    def yyzy(self): return Float4(self.y, self.y, self.z, self.y)
    @property
    def yyzz(self): return Float4(self.y, self.y, self.z, self.z)
    @property
    def yyzw(self): return Float4(self.y, self.y, self.z, self.w)
    @property
    def yywx(self): return Float4(self.y, self.y, self.w, self.x)
    @property
    def yywy(self): return Float4(self.y, self.y, self.w, self.y)
    @property
    def yywz(self): return Float4(self.y, self.y, self.w, self.z)
    @property
    def yyww(self): return Float4(self.y, self.y, self.w, self.w)
    @property
    def yzxx(self): return Float4(self.y, self.z, self.x, self.x)
    @property
    def yzxy(self): return Float4(self.y, self.z, self.x, self.y)
    @property
    def yzxz(self): return Float4(self.y, self.z, self.x, self.z)
    @property
    def yzxw(self): return Float4(self.y, self.z, self.x, self.w)
    @property
    def yzyx(self): return Float4(self.y, self.z, self.y, self.x)
    @property
    def yzyy(self): return Float4(self.y, self.z, self.y, self.y)
    @property
    def yzyz(self): return Float4(self.y, self.z, self.y, self.z)
    @property
    def yzyw(self): return Float4(self.y, self.z, self.y, self.w)
    @property
    def yzzx(self): return Float4(self.y, self.z, self.z, self.x)
    @property
    def yzzy(self): return Float4(self.y, self.z, self.z, self.y)
    @property
    def yzzz(self): return Float4(self.y, self.z, self.z, self.z)
    @property
    def yzzw(self): return Float4(self.y, self.z, self.z, self.w)
    @property
    def yzwx(self): return Float4(self.y, self.z, self.w, self.x)
    @property
    def yzwy(self): return Float4(self.y, self.z, self.w, self.y)
    @property
    def yzwz(self): return Float4(self.y, self.z, self.w, self.z)
    @property
    def yzww(self): return Float4(self.y, self.z, self.w, self.w)
    @property
    def ywxx(self): return Float4(self.y, self.w, self.x, self.x)
    @property
    def ywxy(self): return Float4(self.y, self.w, self.x, self.y)
    @property
    def ywxz(self): return Float4(self.y, self.w, self.x, self.z)
    @property
    def ywxw(self): return Float4(self.y, self.w, self.x, self.w)
    @property
    def ywyx(self): return Float4(self.y, self.w, self.y, self.x)
    @property
    def ywyy(self): return Float4(self.y, self.w, self.y, self.y)
    @property
    def ywyz(self): return Float4(self.y, self.w, self.y, self.z)
    @property
    def ywyw(self): return Float4(self.y, self.w, self.y, self.w)
    @property
    def ywzx(self): return Float4(self.y, self.w, self.z, self.x)
    @property
    def ywzy(self): return Float4(self.y, self.w, self.z, self.y)
    @property
    def ywzz(self): return Float4(self.y, self.w, self.z, self.z)
    @property
    def ywzw(self): return Float4(self.y, self.w, self.z, self.w)
    @property
    def ywwx(self): return Float4(self.y, self.w, self.w, self.x)
    @property
    def ywwy(self): return Float4(self.y, self.w, self.w, self.y)
    @property
    def ywwz(self): return Float4(self.y, self.w, self.w, self.z)
    @property
    def ywww(self): return Float4(self.y, self.w, self.w, self.w)
    @property
    def zxxx(self): return Float4(self.z, self.x, self.x, self.x)
    @property
    def zxxy(self): return Float4(self.z, self.x, self.x, self.y)
    @property
    def zxxz(self): return Float4(self.z, self.x, self.x, self.z)
    @property
    def zxxw(self): return Float4(self.z, self.x, self.x, self.w)
    @property
    def zxyx(self): return Float4(self.z, self.x, self.y, self.x)
    @property
    def zxyy(self): return Float4(self.z, self.x, self.y, self.y)
    @property
    def zxyz(self): return Float4(self.z, self.x, self.y, self.z)
    @property
    def zxyw(self): return Float4(self.z, self.x, self.y, self.w)
    @property
    def zxzx(self): return Float4(self.z, self.x, self.z, self.x)
    @property
    def zxzy(self): return Float4(self.z, self.x, self.z, self.y)
    @property
    def zxzz(self): return Float4(self.z, self.x, self.z, self.z)
    @property
    def zxzw(self): return Float4(self.z, self.x, self.z, self.w)
    @property
    def zxwx(self): return Float4(self.z, self.x, self.w, self.x)
    @property
    def zxwy(self): return Float4(self.z, self.x, self.w, self.y)
    @property
    def zxwz(self): return Float4(self.z, self.x, self.w, self.z)
    @property
    def zxww(self): return Float4(self.z, self.x, self.w, self.w)
    @property
    def zyxx(self): return Float4(self.z, self.y, self.x, self.x)
    @property
    def zyxy(self): return Float4(self.z, self.y, self.x, self.y)
    @property
    def zyxz(self): return Float4(self.z, self.y, self.x, self.z)
    @property
    def zyxw(self): return Float4(self.z, self.y, self.x, self.w)
    @property
    def zyyx(self): return Float4(self.z, self.y, self.y, self.x)
    @property
    def zyyy(self): return Float4(self.z, self.y, self.y, self.y)
    @property
    def zyyz(self): return Float4(self.z, self.y, self.y, self.z)
    @property
    def zyyw(self): return Float4(self.z, self.y, self.y, self.w)
    @property
    def zyzx(self): return Float4(self.z, self.y, self.z, self.x)
    @property
    def zyzy(self): return Float4(self.z, self.y, self.z, self.y)
    @property
    def zyzz(self): return Float4(self.z, self.y, self.z, self.z)
    @property
    def zyzw(self): return Float4(self.z, self.y, self.z, self.w)
    @property
    def zywx(self): return Float4(self.z, self.y, self.w, self.x)
    @property
    def zywy(self): return Float4(self.z, self.y, self.w, self.y)
    @property
    def zywz(self): return Float4(self.z, self.y, self.w, self.z)
    @property
    def zyww(self): return Float4(self.z, self.y, self.w, self.w)
    @property
    def zzxx(self): return Float4(self.z, self.z, self.x, self.x)
    @property
    def zzxy(self): return Float4(self.z, self.z, self.x, self.y)
    @property
    def zzxz(self): return Float4(self.z, self.z, self.x, self.z)
    @property
    def zzxw(self): return Float4(self.z, self.z, self.x, self.w)
    @property
    def zzyx(self): return Float4(self.z, self.z, self.y, self.x)
    @property
    def zzyy(self): return Float4(self.z, self.z, self.y, self.y)
    @property
    def zzyz(self): return Float4(self.z, self.z, self.y, self.z)
    @property
    def zzyw(self): return Float4(self.z, self.z, self.y, self.w)
    @property
    def zzzx(self): return Float4(self.z, self.z, self.z, self.x)
    @property
    def zzzy(self): return Float4(self.z, self.z, self.z, self.y)
    @property
    def zzzz(self): return Float4(self.z, self.z, self.z, self.z)
    @property
    def zzzw(self): return Float4(self.z, self.z, self.z, self.w)
    @property
    def zzwx(self): return Float4(self.z, self.z, self.w, self.x)
    @property
    def zzwy(self): return Float4(self.z, self.z, self.w, self.y)
    @property
    def zzwz(self): return Float4(self.z, self.z, self.w, self.z)
    @property
    def zzww(self): return Float4(self.z, self.z, self.w, self.w)
    @property
    def zwxx(self): return Float4(self.z, self.w, self.x, self.x)
    @property
    def zwxy(self): return Float4(self.z, self.w, self.x, self.y)
    @property
    def zwxz(self): return Float4(self.z, self.w, self.x, self.z)
    @property
    def zwxw(self): return Float4(self.z, self.w, self.x, self.w)
    @property
    def zwyx(self): return Float4(self.z, self.w, self.y, self.x)
    @property
    def zwyy(self): return Float4(self.z, self.w, self.y, self.y)
    @property
    def zwyz(self): return Float4(self.z, self.w, self.y, self.z)
    @property
    def zwyw(self): return Float4(self.z, self.w, self.y, self.w)
    @property
    def zwzx(self): return Float4(self.z, self.w, self.z, self.x)
    @property
    def zwzy(self): return Float4(self.z, self.w, self.z, self.y)
    @property
    def zwzz(self): return Float4(self.z, self.w, self.z, self.z)
    @property
    def zwzw(self): return Float4(self.z, self.w, self.z, self.w)
    @property
    def zwwx(self): return Float4(self.z, self.w, self.w, self.x)
    @property
    def zwwy(self): return Float4(self.z, self.w, self.w, self.y)
    @property
    def zwwz(self): return Float4(self.z, self.w, self.w, self.z)
    @property
    def zwww(self): return Float4(self.z, self.w, self.w, self.w)
    @property
    def wxxx(self): return Float4(self.w, self.x, self.x, self.x)
    @property
    def wxxy(self): return Float4(self.w, self.x, self.x, self.y)
    @property
    def wxxz(self): return Float4(self.w, self.x, self.x, self.z)
    @property
    def wxxw(self): return Float4(self.w, self.x, self.x, self.w)
    @property
    def wxyx(self): return Float4(self.w, self.x, self.y, self.x)
    @property
    def wxyy(self): return Float4(self.w, self.x, self.y, self.y)
    @property
    def wxyz(self): return Float4(self.w, self.x, self.y, self.z)
    @property
    def wxyw(self): return Float4(self.w, self.x, self.y, self.w)
    @property
    def wxzx(self): return Float4(self.w, self.x, self.z, self.x)
    @property
    def wxzy(self): return Float4(self.w, self.x, self.z, self.y)
    @property
    def wxzz(self): return Float4(self.w, self.x, self.z, self.z)
    @property
    def wxzw(self): return Float4(self.w, self.x, self.z, self.w)
    @property
    def wxwx(self): return Float4(self.w, self.x, self.w, self.x)
    @property
    def wxwy(self): return Float4(self.w, self.x, self.w, self.y)
    @property
    def wxwz(self): return Float4(self.w, self.x, self.w, self.z)
    @property
    def wxww(self): return Float4(self.w, self.x, self.w, self.w)
    @property
    def wyxx(self): return Float4(self.w, self.y, self.x, self.x)
    @property
    def wyxy(self): return Float4(self.w, self.y, self.x, self.y)
    @property
    def wyxz(self): return Float4(self.w, self.y, self.x, self.z)
    @property
    def wyxw(self): return Float4(self.w, self.y, self.x, self.w)
    @property
    def wyyx(self): return Float4(self.w, self.y, self.y, self.x)
    @property
    def wyyy(self): return Float4(self.w, self.y, self.y, self.y)
    @property
    def wyyz(self): return Float4(self.w, self.y, self.y, self.z)
    @property
    def wyyw(self): return Float4(self.w, self.y, self.y, self.w)
    @property
    def wyzx(self): return Float4(self.w, self.y, self.z, self.x)
    @property
    def wyzy(self): return Float4(self.w, self.y, self.z, self.y)
    @property
    def wyzz(self): return Float4(self.w, self.y, self.z, self.z)
    @property
    def wyzw(self): return Float4(self.w, self.y, self.z, self.w)
    @property
    def wywx(self): return Float4(self.w, self.y, self.w, self.x)
    @property
    def wywy(self): return Float4(self.w, self.y, self.w, self.y)
    @property
    def wywz(self): return Float4(self.w, self.y, self.w, self.z)
    @property
    def wyww(self): return Float4(self.w, self.y, self.w, self.w)
    @property
    def wzxx(self): return Float4(self.w, self.z, self.x, self.x)
    @property
    def wzxy(self): return Float4(self.w, self.z, self.x, self.y)
    @property
    def wzxz(self): return Float4(self.w, self.z, self.x, self.z)
    @property
    def wzxw(self): return Float4(self.w, self.z, self.x, self.w)
    @property
    def wzyx(self): return Float4(self.w, self.z, self.y, self.x)
    @property
    def wzyy(self): return Float4(self.w, self.z, self.y, self.y)
    @property
    def wzyz(self): return Float4(self.w, self.z, self.y, self.z)
    @property
    def wzyw(self): return Float4(self.w, self.z, self.y, self.w)
    @property
    def wzzx(self): return Float4(self.w, self.z, self.z, self.x)
    @property
    def wzzy(self): return Float4(self.w, self.z, self.z, self.y)
    @property
    def wzzz(self): return Float4(self.w, self.z, self.z, self.z)
    @property
    def wzzw(self): return Float4(self.w, self.z, self.z, self.w)
    @property
    def wzwx(self): return Float4(self.w, self.z, self.w, self.x)
    @property
    def wzwy(self): return Float4(self.w, self.z, self.w, self.y)
    @property
    def wzwz(self): return Float4(self.w, self.z, self.w, self.z)
    @property
    def wzww(self): return Float4(self.w, self.z, self.w, self.w)
    @property
    def wwxx(self): return Float4(self.w, self.w, self.x, self.x)
    @property
    def wwxy(self): return Float4(self.w, self.w, self.x, self.y)
    @property
    def wwxz(self): return Float4(self.w, self.w, self.x, self.z)
    @property
    def wwxw(self): return Float4(self.w, self.w, self.x, self.w)
    @property
    def wwyx(self): return Float4(self.w, self.w, self.y, self.x)
    @property
    def wwyy(self): return Float4(self.w, self.w, self.y, self.y)
    @property
    def wwyz(self): return Float4(self.w, self.w, self.y, self.z)
    @property
    def wwyw(self): return Float4(self.w, self.w, self.y, self.w)
    @property
    def wwzx(self): return Float4(self.w, self.w, self.z, self.x)
    @property
    def wwzy(self): return Float4(self.w, self.w, self.z, self.y)
    @property
    def wwzz(self): return Float4(self.w, self.w, self.z, self.z)
    @property
    def wwzw(self): return Float4(self.w, self.w, self.z, self.w)
    @property
    def wwwx(self): return Float4(self.w, self.w, self.w, self.x)
    @property
    def wwwy(self): return Float4(self.w, self.w, self.w, self.y)
    @property
    def wwwz(self): return Float4(self.w, self.w, self.w, self.z)
    @property
    def wwww(self): return Float4(self.w, self.w, self.w, self.w)

    @property
    def rrrr(self): return Float4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Float4(self.r, self.r, self.r, self.g)
    @property
    def rrrb(self): return Float4(self.r, self.r, self.r, self.b)
    @property
    def rrra(self): return Float4(self.r, self.r, self.r, self.a)
    @property
    def rrgr(self): return Float4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Float4(self.r, self.r, self.g, self.g)
    @property
    def rrgb(self): return Float4(self.r, self.r, self.g, self.b)
    @property
    def rrga(self): return Float4(self.r, self.r, self.g, self.a)
    @property
    def rrbr(self): return Float4(self.r, self.r, self.b, self.r)
    @property
    def rrbg(self): return Float4(self.r, self.r, self.b, self.g)
    @property
    def rrbb(self): return Float4(self.r, self.r, self.b, self.b)
    @property
    def rrba(self): return Float4(self.r, self.r, self.b, self.a)
    @property
    def rrar(self): return Float4(self.r, self.r, self.a, self.r)
    @property
    def rrag(self): return Float4(self.r, self.r, self.a, self.g)
    @property
    def rrab(self): return Float4(self.r, self.r, self.a, self.b)
    @property
    def rraa(self): return Float4(self.r, self.r, self.a, self.a)
    @property
    def rgrr(self): return Float4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Float4(self.r, self.g, self.r, self.g)
    @property
    def rgrb(self): return Float4(self.r, self.g, self.r, self.b)
    @property
    def rgra(self): return Float4(self.r, self.g, self.r, self.a)
    @property
    def rggr(self): return Float4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Float4(self.r, self.g, self.g, self.g)
    @property
    def rggb(self): return Float4(self.r, self.g, self.g, self.b)
    @property
    def rgga(self): return Float4(self.r, self.g, self.g, self.a)
    @property
    def rgbr(self): return Float4(self.r, self.g, self.b, self.r)
    @property
    def rgbg(self): return Float4(self.r, self.g, self.b, self.g)
    @property
    def rgbb(self): return Float4(self.r, self.g, self.b, self.b)
    @property
    def rgba(self): return self
    @property
    def rgar(self): return Float4(self.r, self.g, self.a, self.r)
    @property
    def rgag(self): return Float4(self.r, self.g, self.a, self.g)
    @property
    def rgab(self): return Float4(self.r, self.g, self.a, self.b)
    @property
    def rgaa(self): return Float4(self.r, self.g, self.a, self.a)
    @property
    def rbrr(self): return Float4(self.r, self.b, self.r, self.r)
    @property
    def rbrg(self): return Float4(self.r, self.b, self.r, self.g)
    @property
    def rbrb(self): return Float4(self.r, self.b, self.r, self.b)
    @property
    def rbra(self): return Float4(self.r, self.b, self.r, self.a)
    @property
    def rbgr(self): return Float4(self.r, self.b, self.g, self.r)
    @property
    def rbgg(self): return Float4(self.r, self.b, self.g, self.g)
    @property
    def rbgb(self): return Float4(self.r, self.b, self.g, self.b)
    @property
    def rbga(self): return Float4(self.r, self.b, self.g, self.a)
    @property
    def rbbr(self): return Float4(self.r, self.b, self.b, self.r)
    @property
    def rbbg(self): return Float4(self.r, self.b, self.b, self.g)
    @property
    def rbbb(self): return Float4(self.r, self.b, self.b, self.b)
    @property
    def rbba(self): return Float4(self.r, self.b, self.b, self.a)
    @property
    def rbar(self): return Float4(self.r, self.b, self.a, self.r)
    @property
    def rbag(self): return Float4(self.r, self.b, self.a, self.g)
    @property
    def rbab(self): return Float4(self.r, self.b, self.a, self.b)
    @property
    def rbaa(self): return Float4(self.r, self.b, self.a, self.a)
    @property
    def rarr(self): return Float4(self.r, self.a, self.r, self.r)
    @property
    def rarg(self): return Float4(self.r, self.a, self.r, self.g)
    @property
    def rarb(self): return Float4(self.r, self.a, self.r, self.b)
    @property
    def rara(self): return Float4(self.r, self.a, self.r, self.a)
    @property
    def ragr(self): return Float4(self.r, self.a, self.g, self.r)
    @property
    def ragg(self): return Float4(self.r, self.a, self.g, self.g)
    @property
    def ragb(self): return Float4(self.r, self.a, self.g, self.b)
    @property
    def raga(self): return Float4(self.r, self.a, self.g, self.a)
    @property
    def rabr(self): return Float4(self.r, self.a, self.b, self.r)
    @property
    def rabg(self): return Float4(self.r, self.a, self.b, self.g)
    @property
    def rabb(self): return Float4(self.r, self.a, self.b, self.b)
    @property
    def raba(self): return Float4(self.r, self.a, self.b, self.a)
    @property
    def raar(self): return Float4(self.r, self.a, self.a, self.r)
    @property
    def raag(self): return Float4(self.r, self.a, self.a, self.g)
    @property
    def raab(self): return Float4(self.r, self.a, self.a, self.b)
    @property
    def raaa(self): return Float4(self.r, self.a, self.a, self.a)
    @property
    def grrr(self): return Float4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Float4(self.g, self.r, self.r, self.g)
    @property
    def grrb(self): return Float4(self.g, self.r, self.r, self.b)
    @property
    def grra(self): return Float4(self.g, self.r, self.r, self.a)
    @property
    def grgr(self): return Float4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Float4(self.g, self.r, self.g, self.g)
    @property
    def grgb(self): return Float4(self.g, self.r, self.g, self.b)
    @property
    def grga(self): return Float4(self.g, self.r, self.g, self.a)
    @property
    def grbr(self): return Float4(self.g, self.r, self.b, self.r)
    @property
    def grbg(self): return Float4(self.g, self.r, self.b, self.g)
    @property
    def grbb(self): return Float4(self.g, self.r, self.b, self.b)
    @property
    def grba(self): return Float4(self.g, self.r, self.b, self.a)
    @property
    def grar(self): return Float4(self.g, self.r, self.a, self.r)
    @property
    def grag(self): return Float4(self.g, self.r, self.a, self.g)
    @property
    def grab(self): return Float4(self.g, self.r, self.a, self.b)
    @property
    def graa(self): return Float4(self.g, self.r, self.a, self.a)
    @property
    def ggrr(self): return Float4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Float4(self.g, self.g, self.r, self.g)
    @property
    def ggrb(self): return Float4(self.g, self.g, self.r, self.b)
    @property
    def ggra(self): return Float4(self.g, self.g, self.r, self.a)
    @property
    def gggr(self): return Float4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Float4(self.g, self.g, self.g, self.g)
    @property
    def gggb(self): return Float4(self.g, self.g, self.g, self.b)
    @property
    def ggga(self): return Float4(self.g, self.g, self.g, self.a)
    @property
    def ggbr(self): return Float4(self.g, self.g, self.b, self.r)
    @property
    def ggbg(self): return Float4(self.g, self.g, self.b, self.g)
    @property
    def ggbb(self): return Float4(self.g, self.g, self.b, self.b)
    @property
    def ggba(self): return Float4(self.g, self.g, self.b, self.a)
    @property
    def ggar(self): return Float4(self.g, self.g, self.a, self.r)
    @property
    def ggag(self): return Float4(self.g, self.g, self.a, self.g)
    @property
    def ggab(self): return Float4(self.g, self.g, self.a, self.b)
    @property
    def ggaa(self): return Float4(self.g, self.g, self.a, self.a)
    @property
    def gbrr(self): return Float4(self.g, self.b, self.r, self.r)
    @property
    def gbrg(self): return Float4(self.g, self.b, self.r, self.g)
    @property
    def gbrb(self): return Float4(self.g, self.b, self.r, self.b)
    @property
    def gbra(self): return Float4(self.g, self.b, self.r, self.a)
    @property
    def gbgr(self): return Float4(self.g, self.b, self.g, self.r)
    @property
    def gbgg(self): return Float4(self.g, self.b, self.g, self.g)
    @property
    def gbgb(self): return Float4(self.g, self.b, self.g, self.b)
    @property
    def gbga(self): return Float4(self.g, self.b, self.g, self.a)
    @property
    def gbbr(self): return Float4(self.g, self.b, self.b, self.r)
    @property
    def gbbg(self): return Float4(self.g, self.b, self.b, self.g)
    @property
    def gbbb(self): return Float4(self.g, self.b, self.b, self.b)
    @property
    def gbba(self): return Float4(self.g, self.b, self.b, self.a)
    @property
    def gbar(self): return Float4(self.g, self.b, self.a, self.r)
    @property
    def gbag(self): return Float4(self.g, self.b, self.a, self.g)
    @property
    def gbab(self): return Float4(self.g, self.b, self.a, self.b)
    @property
    def gbaa(self): return Float4(self.g, self.b, self.a, self.a)
    @property
    def garr(self): return Float4(self.g, self.a, self.r, self.r)
    @property
    def garg(self): return Float4(self.g, self.a, self.r, self.g)
    @property
    def garb(self): return Float4(self.g, self.a, self.r, self.b)
    @property
    def gara(self): return Float4(self.g, self.a, self.r, self.a)
    @property
    def gagr(self): return Float4(self.g, self.a, self.g, self.r)
    @property
    def gagg(self): return Float4(self.g, self.a, self.g, self.g)
    @property
    def gagb(self): return Float4(self.g, self.a, self.g, self.b)
    @property
    def gaga(self): return Float4(self.g, self.a, self.g, self.a)
    @property
    def gabr(self): return Float4(self.g, self.a, self.b, self.r)
    @property
    def gabg(self): return Float4(self.g, self.a, self.b, self.g)
    @property
    def gabb(self): return Float4(self.g, self.a, self.b, self.b)
    @property
    def gaba(self): return Float4(self.g, self.a, self.b, self.a)
    @property
    def gaar(self): return Float4(self.g, self.a, self.a, self.r)
    @property
    def gaag(self): return Float4(self.g, self.a, self.a, self.g)
    @property
    def gaab(self): return Float4(self.g, self.a, self.a, self.b)
    @property
    def gaaa(self): return Float4(self.g, self.a, self.a, self.a)
    @property
    def brrr(self): return Float4(self.b, self.r, self.r, self.r)
    @property
    def brrg(self): return Float4(self.b, self.r, self.r, self.g)
    @property
    def brrb(self): return Float4(self.b, self.r, self.r, self.b)
    @property
    def brra(self): return Float4(self.b, self.r, self.r, self.a)
    @property
    def brgr(self): return Float4(self.b, self.r, self.g, self.r)
    @property
    def brgg(self): return Float4(self.b, self.r, self.g, self.g)
    @property
    def brgb(self): return Float4(self.b, self.r, self.g, self.b)
    @property
    def brga(self): return Float4(self.b, self.r, self.g, self.a)
    @property
    def brbr(self): return Float4(self.b, self.r, self.b, self.r)
    @property
    def brbg(self): return Float4(self.b, self.r, self.b, self.g)
    @property
    def brbb(self): return Float4(self.b, self.r, self.b, self.b)
    @property
    def brba(self): return Float4(self.b, self.r, self.b, self.a)
    @property
    def brar(self): return Float4(self.b, self.r, self.a, self.r)
    @property
    def brag(self): return Float4(self.b, self.r, self.a, self.g)
    @property
    def brab(self): return Float4(self.b, self.r, self.a, self.b)
    @property
    def braa(self): return Float4(self.b, self.r, self.a, self.a)
    @property
    def bgrr(self): return Float4(self.b, self.g, self.r, self.r)
    @property
    def bgrg(self): return Float4(self.b, self.g, self.r, self.g)
    @property
    def bgrb(self): return Float4(self.b, self.g, self.r, self.b)
    @property
    def bgra(self): return Float4(self.b, self.g, self.r, self.a)
    @property
    def bggr(self): return Float4(self.b, self.g, self.g, self.r)
    @property
    def bggg(self): return Float4(self.b, self.g, self.g, self.g)
    @property
    def bggb(self): return Float4(self.b, self.g, self.g, self.b)
    @property
    def bgga(self): return Float4(self.b, self.g, self.g, self.a)
    @property
    def bgbr(self): return Float4(self.b, self.g, self.b, self.r)
    @property
    def bgbg(self): return Float4(self.b, self.g, self.b, self.g)
    @property
    def bgbb(self): return Float4(self.b, self.g, self.b, self.b)
    @property
    def bgba(self): return Float4(self.b, self.g, self.b, self.a)
    @property
    def bgar(self): return Float4(self.b, self.g, self.a, self.r)
    @property
    def bgag(self): return Float4(self.b, self.g, self.a, self.g)
    @property
    def bgab(self): return Float4(self.b, self.g, self.a, self.b)
    @property
    def bgaa(self): return Float4(self.b, self.g, self.a, self.a)
    @property
    def bbrr(self): return Float4(self.b, self.b, self.r, self.r)
    @property
    def bbrg(self): return Float4(self.b, self.b, self.r, self.g)
    @property
    def bbrb(self): return Float4(self.b, self.b, self.r, self.b)
    @property
    def bbra(self): return Float4(self.b, self.b, self.r, self.a)
    @property
    def bbgr(self): return Float4(self.b, self.b, self.g, self.r)
    @property
    def bbgg(self): return Float4(self.b, self.b, self.g, self.g)
    @property
    def bbgb(self): return Float4(self.b, self.b, self.g, self.b)
    @property
    def bbga(self): return Float4(self.b, self.b, self.g, self.a)
    @property
    def bbbr(self): return Float4(self.b, self.b, self.b, self.r)
    @property
    def bbbg(self): return Float4(self.b, self.b, self.b, self.g)
    @property
    def bbbb(self): return Float4(self.b, self.b, self.b, self.b)
    @property
    def bbba(self): return Float4(self.b, self.b, self.b, self.a)
    @property
    def bbar(self): return Float4(self.b, self.b, self.a, self.r)
    @property
    def bbag(self): return Float4(self.b, self.b, self.a, self.g)
    @property
    def bbab(self): return Float4(self.b, self.b, self.a, self.b)
    @property
    def bbaa(self): return Float4(self.b, self.b, self.a, self.a)
    @property
    def barr(self): return Float4(self.b, self.a, self.r, self.r)
    @property
    def barg(self): return Float4(self.b, self.a, self.r, self.g)
    @property
    def barb(self): return Float4(self.b, self.a, self.r, self.b)
    @property
    def bara(self): return Float4(self.b, self.a, self.r, self.a)
    @property
    def bagr(self): return Float4(self.b, self.a, self.g, self.r)
    @property
    def bagg(self): return Float4(self.b, self.a, self.g, self.g)
    @property
    def bagb(self): return Float4(self.b, self.a, self.g, self.b)
    @property
    def baga(self): return Float4(self.b, self.a, self.g, self.a)
    @property
    def babr(self): return Float4(self.b, self.a, self.b, self.r)
    @property
    def babg(self): return Float4(self.b, self.a, self.b, self.g)
    @property
    def babb(self): return Float4(self.b, self.a, self.b, self.b)
    @property
    def baba(self): return Float4(self.b, self.a, self.b, self.a)
    @property
    def baar(self): return Float4(self.b, self.a, self.a, self.r)
    @property
    def baag(self): return Float4(self.b, self.a, self.a, self.g)
    @property
    def baab(self): return Float4(self.b, self.a, self.a, self.b)
    @property
    def baaa(self): return Float4(self.b, self.a, self.a, self.a)
    @property
    def arrr(self): return Float4(self.a, self.r, self.r, self.r)
    @property
    def arrg(self): return Float4(self.a, self.r, self.r, self.g)
    @property
    def arrb(self): return Float4(self.a, self.r, self.r, self.b)
    @property
    def arra(self): return Float4(self.a, self.r, self.r, self.a)
    @property
    def argr(self): return Float4(self.a, self.r, self.g, self.r)
    @property
    def argg(self): return Float4(self.a, self.r, self.g, self.g)
    @property
    def argb(self): return Float4(self.a, self.r, self.g, self.b)
    @property
    def arga(self): return Float4(self.a, self.r, self.g, self.a)
    @property
    def arbr(self): return Float4(self.a, self.r, self.b, self.r)
    @property
    def arbg(self): return Float4(self.a, self.r, self.b, self.g)
    @property
    def arbb(self): return Float4(self.a, self.r, self.b, self.b)
    @property
    def arba(self): return Float4(self.a, self.r, self.b, self.a)
    @property
    def arar(self): return Float4(self.a, self.r, self.a, self.r)
    @property
    def arag(self): return Float4(self.a, self.r, self.a, self.g)
    @property
    def arab(self): return Float4(self.a, self.r, self.a, self.b)
    @property
    def araa(self): return Float4(self.a, self.r, self.a, self.a)
    @property
    def agrr(self): return Float4(self.a, self.g, self.r, self.r)
    @property
    def agrg(self): return Float4(self.a, self.g, self.r, self.g)
    @property
    def agrb(self): return Float4(self.a, self.g, self.r, self.b)
    @property
    def agra(self): return Float4(self.a, self.g, self.r, self.a)
    @property
    def aggr(self): return Float4(self.a, self.g, self.g, self.r)
    @property
    def aggg(self): return Float4(self.a, self.g, self.g, self.g)
    @property
    def aggb(self): return Float4(self.a, self.g, self.g, self.b)
    @property
    def agga(self): return Float4(self.a, self.g, self.g, self.a)
    @property
    def agbr(self): return Float4(self.a, self.g, self.b, self.r)
    @property
    def agbg(self): return Float4(self.a, self.g, self.b, self.g)
    @property
    def agbb(self): return Float4(self.a, self.g, self.b, self.b)
    @property
    def agba(self): return Float4(self.a, self.g, self.b, self.a)
    @property
    def agar(self): return Float4(self.a, self.g, self.a, self.r)
    @property
    def agag(self): return Float4(self.a, self.g, self.a, self.g)
    @property
    def agab(self): return Float4(self.a, self.g, self.a, self.b)
    @property
    def agaa(self): return Float4(self.a, self.g, self.a, self.a)
    @property
    def abrr(self): return Float4(self.a, self.b, self.r, self.r)
    @property
    def abrg(self): return Float4(self.a, self.b, self.r, self.g)
    @property
    def abrb(self): return Float4(self.a, self.b, self.r, self.b)
    @property
    def abra(self): return Float4(self.a, self.b, self.r, self.a)
    @property
    def abgr(self): return Float4(self.a, self.b, self.g, self.r)
    @property
    def abgg(self): return Float4(self.a, self.b, self.g, self.g)
    @property
    def abgb(self): return Float4(self.a, self.b, self.g, self.b)
    @property
    def abga(self): return Float4(self.a, self.b, self.g, self.a)
    @property
    def abbr(self): return Float4(self.a, self.b, self.b, self.r)
    @property
    def abbg(self): return Float4(self.a, self.b, self.b, self.g)
    @property
    def abbb(self): return Float4(self.a, self.b, self.b, self.b)
    @property
    def abba(self): return Float4(self.a, self.b, self.b, self.a)
    @property
    def abar(self): return Float4(self.a, self.b, self.a, self.r)
    @property
    def abag(self): return Float4(self.a, self.b, self.a, self.g)
    @property
    def abab(self): return Float4(self.a, self.b, self.a, self.b)
    @property
    def abaa(self): return Float4(self.a, self.b, self.a, self.a)
    @property
    def aarr(self): return Float4(self.a, self.a, self.r, self.r)
    @property
    def aarg(self): return Float4(self.a, self.a, self.r, self.g)
    @property
    def aarb(self): return Float4(self.a, self.a, self.r, self.b)
    @property
    def aara(self): return Float4(self.a, self.a, self.r, self.a)
    @property
    def aagr(self): return Float4(self.a, self.a, self.g, self.r)
    @property
    def aagg(self): return Float4(self.a, self.a, self.g, self.g)
    @property
    def aagb(self): return Float4(self.a, self.a, self.g, self.b)
    @property
    def aaga(self): return Float4(self.a, self.a, self.g, self.a)
    @property
    def aabr(self): return Float4(self.a, self.a, self.b, self.r)
    @property
    def aabg(self): return Float4(self.a, self.a, self.b, self.g)
    @property
    def aabb(self): return Float4(self.a, self.a, self.b, self.b)
    @property
    def aaba(self): return Float4(self.a, self.a, self.b, self.a)
    @property
    def aaar(self): return Float4(self.a, self.a, self.a, self.r)
    @property
    def aaag(self): return Float4(self.a, self.a, self.a, self.g)
    @property
    def aaab(self): return Float4(self.a, self.a, self.a, self.b)
    @property
    def aaaa(self): return Float4(self.a, self.a, self.a, self.a)

class Int2(np.ndarray):
    """
    Int2 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 2, "list/tuple must have 2 components"
                arr = np.asarray([args[0][0], args[0][1]], dtype=np.int32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 2, \
                    "numpy array must be sized [C=2] or [C=2, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.int32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 2, \
                    "torch tensor must be sized [C=2] or [C=2, H=1, W=1]"
                value = args[0].squeeze().int().cpu()
                arr = np.asarray([value[0].item(), value[1].item()], dtype=np.int32).view(cls)
            else:
                value = int(args[0])
                arr = np.asarray([value, value], dtype=np.int32).view(cls)
        elif len(args) == 2:
            arr = np.asarray(args, dtype=np.int32).view(cls)
        else: 
            raise TypeError("Int2 only accepts 1 or 2 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1])

    @property
    def x(self) -> int:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> int:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def r(self) -> int:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> int:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value

    @property
    def xx(self): return Int2(self.x, self.x)
    @property
    def xy(self): return self
    @property
    def yx(self): return Int2(self.y, self.x)
    @property
    def yy(self): return Int2(self.y, self.y)

    @property
    def rr(self): return Int2(self.r, self.r)
    @property
    def rg(self): return self
    @property
    def gr(self): return Int2(self.g, self.r)
    @property
    def gg(self): return Int2(self.g, self.g)

    @property
    def xxx(self): return Int3(self.x, self.x, self.x)
    @property
    def xxy(self): return Int3(self.x, self.x, self.y)
    @property
    def xyx(self): return Int3(self.x, self.y, self.x)
    @property
    def xyy(self): return Int3(self.x, self.y, self.y)
    @property
    def yxx(self): return Int3(self.y, self.x, self.x)
    @property
    def yxy(self): return Int3(self.y, self.x, self.y)
    @property
    def yyx(self): return Int3(self.y, self.y, self.x)
    @property
    def yyy(self): return Int3(self.y, self.y, self.y)

    @property
    def rrr(self): return Int3(self.r, self.r, self.r)
    @property
    def rrg(self): return Int3(self.r, self.r, self.g)
    @property
    def rgr(self): return Int3(self.r, self.g, self.r)
    @property
    def rgg(self): return Int3(self.r, self.g, self.g)
    @property
    def grr(self): return Int3(self.g, self.r, self.r)
    @property
    def grg(self): return Int3(self.g, self.r, self.g)
    @property
    def ggr(self): return Int3(self.g, self.g, self.r)
    @property
    def ggg(self): return Int3(self.g, self.g, self.g)

    @property
    def xxxx(self): return Int4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Int4(self.x, self.x, self.x, self.y)
    @property
    def xxyx(self): return Int4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Int4(self.x, self.x, self.y, self.y)
    @property
    def xyxx(self): return Int4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Int4(self.x, self.y, self.x, self.y)
    @property
    def xyyx(self): return Int4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Int4(self.x, self.y, self.y, self.y)
    @property
    def yxxx(self): return Int4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Int4(self.y, self.x, self.x, self.y)
    @property
    def yxyx(self): return Int4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Int4(self.y, self.x, self.y, self.y)
    @property
    def yyxx(self): return Int4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Int4(self.y, self.y, self.x, self.y)
    @property
    def yyyx(self): return Int4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Int4(self.y, self.y, self.y, self.y)

    @property
    def rrrr(self): return Int4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Int4(self.r, self.r, self.r, self.g)
    @property
    def rrgr(self): return Int4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Int4(self.r, self.r, self.g, self.g)
    @property
    def rgrr(self): return Int4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Int4(self.r, self.g, self.r, self.g)
    @property
    def rggr(self): return Int4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Int4(self.r, self.g, self.g, self.g)
    @property
    def grrr(self): return Int4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Int4(self.g, self.r, self.r, self.g)
    @property
    def grgr(self): return Int4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Int4(self.g, self.r, self.g, self.g)
    @property
    def ggrr(self): return Int4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Int4(self.g, self.g, self.r, self.g)
    @property
    def gggr(self): return Int4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Int4(self.g, self.g, self.g, self.g)


class Int3(np.ndarray):
    """
    Int3 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 3, "list/tuple must have 3 components"
                arr = np.asarray([args[0][0], args[0][1], args[0][2]], dtype=np.int32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 3, \
                    "numpy array must be sized [C=3] or [C=3, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.int32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 3, \
                    "torch tensor must be sized [C=3] or [C=3, H=1, W=1]"
                value = args[0].squeeze().int().cpu()
                arr = np.asarray([value[0].item(), value[1].item(), value[2].item()], dtype=np.int32).view(cls)
            else:
                value = int(args[0])
                arr = np.asarray([value, value, value], dtype=np.int32).view(cls)
        elif len(args) == 3:
            arr = np.asarray(args, dtype=np.int32).view(cls)
        else: 
            raise TypeError("Int3 only accepts 1 or 3 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1], self[2]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1], self[2])

    @property
    def x(self) -> int:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> int:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def z(self) -> int:
        return self[2]
    @z.setter
    def z(self, value):
        self[2] = value
    @property
    def r(self) -> int:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> int:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value
    @property
    def b(self) -> int:
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value

    @property
    def xx(self): return Int2(self.x, self.x)
    @property
    def xy(self): return Int2(self.x, self.y)
    @property
    def xz(self): return Int2(self.x, self.z)
    @property
    def yx(self): return Int2(self.y, self.x)
    @property
    def yy(self): return Int2(self.y, self.y)
    @property
    def yz(self): return Int2(self.y, self.z)
    @property
    def zx(self): return Int2(self.z, self.x)
    @property
    def zy(self): return Int2(self.z, self.y)
    @property
    def zz(self): return Int2(self.z, self.z)

    @property
    def rr(self): return Int2(self.r, self.r)
    @property
    def rg(self): return Int2(self.r, self.g)
    @property
    def rb(self): return Int2(self.r, self.b)
    @property
    def gr(self): return Int2(self.g, self.r)
    @property
    def gg(self): return Int2(self.g, self.g)
    @property
    def gb(self): return Int2(self.g, self.b)
    @property
    def br(self): return Int2(self.b, self.r)
    @property
    def bg(self): return Int2(self.b, self.g)
    @property
    def bb(self): return Int2(self.b, self.b)

    @property
    def xxx(self): return Int3(self.x, self.x, self.x)
    @property
    def xxy(self): return Int3(self.x, self.x, self.y)
    @property
    def xxz(self): return Int3(self.x, self.x, self.z)
    @property
    def xyx(self): return Int3(self.x, self.y, self.x)
    @property
    def xyy(self): return Int3(self.x, self.y, self.y)
    @property
    def xyz(self): return self
    @property
    def xzx(self): return Int3(self.x, self.z, self.x)
    @property
    def xzy(self): return Int3(self.x, self.z, self.y)
    @property
    def xzz(self): return Int3(self.x, self.z, self.z)
    @property
    def yxx(self): return Int3(self.y, self.x, self.x)
    @property
    def yxy(self): return Int3(self.y, self.x, self.y)
    @property
    def yxz(self): return Int3(self.y, self.x, self.z)
    @property
    def yyx(self): return Int3(self.y, self.y, self.x)
    @property
    def yyy(self): return Int3(self.y, self.y, self.y)
    @property
    def yyz(self): return Int3(self.y, self.y, self.z)
    @property
    def yzx(self): return Int3(self.y, self.z, self.x)
    @property
    def yzy(self): return Int3(self.y, self.z, self.y)
    @property
    def yzz(self): return Int3(self.y, self.z, self.z)
    @property
    def zxx(self): return Int3(self.z, self.x, self.x)
    @property
    def zxy(self): return Int3(self.z, self.x, self.y)
    @property
    def zxz(self): return Int3(self.z, self.x, self.z)
    @property
    def zyx(self): return Int3(self.z, self.y, self.x)
    @property
    def zyy(self): return Int3(self.z, self.y, self.y)
    @property
    def zyz(self): return Int3(self.z, self.y, self.z)
    @property
    def zzx(self): return Int3(self.z, self.z, self.x)
    @property
    def zzy(self): return Int3(self.z, self.z, self.y)
    @property
    def zzz(self): return Int3(self.z, self.z, self.z)

    @property
    def rrr(self): return Int3(self.r, self.r, self.r)
    @property
    def rrg(self): return Int3(self.r, self.r, self.g)
    @property
    def rrb(self): return Int3(self.r, self.r, self.b)
    @property
    def rgr(self): return Int3(self.r, self.g, self.r)
    @property
    def rgg(self): return Int3(self.r, self.g, self.g)
    @property
    def rgb(self): return self
    @property
    def rbr(self): return Int3(self.r, self.b, self.r)
    @property
    def rbg(self): return Int3(self.r, self.b, self.g)
    @property
    def rbb(self): return Int3(self.r, self.b, self.b)
    @property
    def grr(self): return Int3(self.g, self.r, self.r)
    @property
    def grg(self): return Int3(self.g, self.r, self.g)
    @property
    def grb(self): return Int3(self.g, self.r, self.b)
    @property
    def ggr(self): return Int3(self.g, self.g, self.r)
    @property
    def ggg(self): return Int3(self.g, self.g, self.g)
    @property
    def ggb(self): return Int3(self.g, self.g, self.b)
    @property
    def gbr(self): return Int3(self.g, self.b, self.r)
    @property
    def gbg(self): return Int3(self.g, self.b, self.g)
    @property
    def gbb(self): return Int3(self.g, self.b, self.b)
    @property
    def brr(self): return Int3(self.b, self.r, self.r)
    @property
    def brg(self): return Int3(self.b, self.r, self.g)
    @property
    def brb(self): return Int3(self.b, self.r, self.b)
    @property
    def bgr(self): return Int3(self.b, self.g, self.r)
    @property
    def bgg(self): return Int3(self.b, self.g, self.g)
    @property
    def bgb(self): return Int3(self.b, self.g, self.b)
    @property
    def bbr(self): return Int3(self.b, self.b, self.r)
    @property
    def bbg(self): return Int3(self.b, self.b, self.g)
    @property
    def bbb(self): return Int3(self.b, self.b, self.b)

    @property
    def xxxx(self): return Int4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Int4(self.x, self.x, self.x, self.y)
    @property
    def xxxz(self): return Int4(self.x, self.x, self.x, self.z)
    @property
    def xxyx(self): return Int4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Int4(self.x, self.x, self.y, self.y)
    @property
    def xxyz(self): return Int4(self.x, self.x, self.y, self.z)
    @property
    def xxzx(self): return Int4(self.x, self.x, self.z, self.x)
    @property
    def xxzy(self): return Int4(self.x, self.x, self.z, self.y)
    @property
    def xxzz(self): return Int4(self.x, self.x, self.z, self.z)
    @property
    def xyxx(self): return Int4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Int4(self.x, self.y, self.x, self.y)
    @property
    def xyxz(self): return Int4(self.x, self.y, self.x, self.z)
    @property
    def xyyx(self): return Int4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Int4(self.x, self.y, self.y, self.y)
    @property
    def xyyz(self): return Int4(self.x, self.y, self.y, self.z)
    @property
    def xyzx(self): return Int4(self.x, self.y, self.z, self.x)
    @property
    def xyzy(self): return Int4(self.x, self.y, self.z, self.y)
    @property
    def xyzz(self): return Int4(self.x, self.y, self.z, self.z)
    @property
    def xzxx(self): return Int4(self.x, self.z, self.x, self.x)
    @property
    def xzxy(self): return Int4(self.x, self.z, self.x, self.y)
    @property
    def xzxz(self): return Int4(self.x, self.z, self.x, self.z)
    @property
    def xzyx(self): return Int4(self.x, self.z, self.y, self.x)
    @property
    def xzyy(self): return Int4(self.x, self.z, self.y, self.y)
    @property
    def xzyz(self): return Int4(self.x, self.z, self.y, self.z)
    @property
    def xzzx(self): return Int4(self.x, self.z, self.z, self.x)
    @property
    def xzzy(self): return Int4(self.x, self.z, self.z, self.y)
    @property
    def xzzz(self): return Int4(self.x, self.z, self.z, self.z)
    @property
    def yxxx(self): return Int4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Int4(self.y, self.x, self.x, self.y)
    @property
    def yxxz(self): return Int4(self.y, self.x, self.x, self.z)
    @property
    def yxyx(self): return Int4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Int4(self.y, self.x, self.y, self.y)
    @property
    def yxyz(self): return Int4(self.y, self.x, self.y, self.z)
    @property
    def yxzx(self): return Int4(self.y, self.x, self.z, self.x)
    @property
    def yxzy(self): return Int4(self.y, self.x, self.z, self.y)
    @property
    def yxzz(self): return Int4(self.y, self.x, self.z, self.z)
    @property
    def yyxx(self): return Int4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Int4(self.y, self.y, self.x, self.y)
    @property
    def yyxz(self): return Int4(self.y, self.y, self.x, self.z)
    @property
    def yyyx(self): return Int4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Int4(self.y, self.y, self.y, self.y)
    @property
    def yyyz(self): return Int4(self.y, self.y, self.y, self.z)
    @property
    def yyzx(self): return Int4(self.y, self.y, self.z, self.x)
    @property
    def yyzy(self): return Int4(self.y, self.y, self.z, self.y)
    @property
    def yyzz(self): return Int4(self.y, self.y, self.z, self.z)
    @property
    def yzxx(self): return Int4(self.y, self.z, self.x, self.x)
    @property
    def yzxy(self): return Int4(self.y, self.z, self.x, self.y)
    @property
    def yzxz(self): return Int4(self.y, self.z, self.x, self.z)
    @property
    def yzyx(self): return Int4(self.y, self.z, self.y, self.x)
    @property
    def yzyy(self): return Int4(self.y, self.z, self.y, self.y)
    @property
    def yzyz(self): return Int4(self.y, self.z, self.y, self.z)
    @property
    def yzzx(self): return Int4(self.y, self.z, self.z, self.x)
    @property
    def yzzy(self): return Int4(self.y, self.z, self.z, self.y)
    @property
    def yzzz(self): return Int4(self.y, self.z, self.z, self.z)
    @property
    def zxxx(self): return Int4(self.z, self.x, self.x, self.x)
    @property
    def zxxy(self): return Int4(self.z, self.x, self.x, self.y)
    @property
    def zxxz(self): return Int4(self.z, self.x, self.x, self.z)
    @property
    def zxyx(self): return Int4(self.z, self.x, self.y, self.x)
    @property
    def zxyy(self): return Int4(self.z, self.x, self.y, self.y)
    @property
    def zxyz(self): return Int4(self.z, self.x, self.y, self.z)
    @property
    def zxzx(self): return Int4(self.z, self.x, self.z, self.x)
    @property
    def zxzy(self): return Int4(self.z, self.x, self.z, self.y)
    @property
    def zxzz(self): return Int4(self.z, self.x, self.z, self.z)
    @property
    def zyxx(self): return Int4(self.z, self.y, self.x, self.x)
    @property
    def zyxy(self): return Int4(self.z, self.y, self.x, self.y)
    @property
    def zyxz(self): return Int4(self.z, self.y, self.x, self.z)
    @property
    def zyyx(self): return Int4(self.z, self.y, self.y, self.x)
    @property
    def zyyy(self): return Int4(self.z, self.y, self.y, self.y)
    @property
    def zyyz(self): return Int4(self.z, self.y, self.y, self.z)
    @property
    def zyzx(self): return Int4(self.z, self.y, self.z, self.x)
    @property
    def zyzy(self): return Int4(self.z, self.y, self.z, self.y)
    @property
    def zyzz(self): return Int4(self.z, self.y, self.z, self.z)
    @property
    def zzxx(self): return Int4(self.z, self.z, self.x, self.x)
    @property
    def zzxy(self): return Int4(self.z, self.z, self.x, self.y)
    @property
    def zzxz(self): return Int4(self.z, self.z, self.x, self.z)
    @property
    def zzyx(self): return Int4(self.z, self.z, self.y, self.x)
    @property
    def zzyy(self): return Int4(self.z, self.z, self.y, self.y)
    @property
    def zzyz(self): return Int4(self.z, self.z, self.y, self.z)
    @property
    def zzzx(self): return Int4(self.z, self.z, self.z, self.x)
    @property
    def zzzy(self): return Int4(self.z, self.z, self.z, self.y)
    @property
    def zzzz(self): return Int4(self.z, self.z, self.z, self.z)

    @property
    def rrrr(self): return Int4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Int4(self.r, self.r, self.r, self.g)
    @property
    def rrrb(self): return Int4(self.r, self.r, self.r, self.b)
    @property
    def rrgr(self): return Int4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Int4(self.r, self.r, self.g, self.g)
    @property
    def rrgb(self): return Int4(self.r, self.r, self.g, self.b)
    @property
    def rrbr(self): return Int4(self.r, self.r, self.b, self.r)
    @property
    def rrbg(self): return Int4(self.r, self.r, self.b, self.g)
    @property
    def rrbb(self): return Int4(self.r, self.r, self.b, self.b)
    @property
    def rgrr(self): return Int4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Int4(self.r, self.g, self.r, self.g)
    @property
    def rgrb(self): return Int4(self.r, self.g, self.r, self.b)
    @property
    def rggr(self): return Int4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Int4(self.r, self.g, self.g, self.g)
    @property
    def rggb(self): return Int4(self.r, self.g, self.g, self.b)
    @property
    def rgbr(self): return Int4(self.r, self.g, self.b, self.r)
    @property
    def rgbg(self): return Int4(self.r, self.g, self.b, self.g)
    @property
    def rgbb(self): return Int4(self.r, self.g, self.b, self.b)
    @property
    def rbrr(self): return Int4(self.r, self.b, self.r, self.r)
    @property
    def rbrg(self): return Int4(self.r, self.b, self.r, self.g)
    @property
    def rbrb(self): return Int4(self.r, self.b, self.r, self.b)
    @property
    def rbgr(self): return Int4(self.r, self.b, self.g, self.r)
    @property
    def rbgg(self): return Int4(self.r, self.b, self.g, self.g)
    @property
    def rbgb(self): return Int4(self.r, self.b, self.g, self.b)
    @property
    def rbbr(self): return Int4(self.r, self.b, self.b, self.r)
    @property
    def rbbg(self): return Int4(self.r, self.b, self.b, self.g)
    @property
    def rbbb(self): return Int4(self.r, self.b, self.b, self.b)
    @property
    def grrr(self): return Int4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Int4(self.g, self.r, self.r, self.g)
    @property
    def grrb(self): return Int4(self.g, self.r, self.r, self.b)
    @property
    def grgr(self): return Int4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Int4(self.g, self.r, self.g, self.g)
    @property
    def grgb(self): return Int4(self.g, self.r, self.g, self.b)
    @property
    def grbr(self): return Int4(self.g, self.r, self.b, self.r)
    @property
    def grbg(self): return Int4(self.g, self.r, self.b, self.g)
    @property
    def grbb(self): return Int4(self.g, self.r, self.b, self.b)
    @property
    def ggrr(self): return Int4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Int4(self.g, self.g, self.r, self.g)
    @property
    def ggrb(self): return Int4(self.g, self.g, self.r, self.b)
    @property
    def gggr(self): return Int4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Int4(self.g, self.g, self.g, self.g)
    @property
    def gggb(self): return Int4(self.g, self.g, self.g, self.b)
    @property
    def ggbr(self): return Int4(self.g, self.g, self.b, self.r)
    @property
    def ggbg(self): return Int4(self.g, self.g, self.b, self.g)
    @property
    def ggbb(self): return Int4(self.g, self.g, self.b, self.b)
    @property
    def gbrr(self): return Int4(self.g, self.b, self.r, self.r)
    @property
    def gbrg(self): return Int4(self.g, self.b, self.r, self.g)
    @property
    def gbrb(self): return Int4(self.g, self.b, self.r, self.b)
    @property
    def gbgr(self): return Int4(self.g, self.b, self.g, self.r)
    @property
    def gbgg(self): return Int4(self.g, self.b, self.g, self.g)
    @property
    def gbgb(self): return Int4(self.g, self.b, self.g, self.b)
    @property
    def gbbr(self): return Int4(self.g, self.b, self.b, self.r)
    @property
    def gbbg(self): return Int4(self.g, self.b, self.b, self.g)
    @property
    def gbbb(self): return Int4(self.g, self.b, self.b, self.b)
    @property
    def brrr(self): return Int4(self.b, self.r, self.r, self.r)
    @property
    def brrg(self): return Int4(self.b, self.r, self.r, self.g)
    @property
    def brrb(self): return Int4(self.b, self.r, self.r, self.b)
    @property
    def brgr(self): return Int4(self.b, self.r, self.g, self.r)
    @property
    def brgg(self): return Int4(self.b, self.r, self.g, self.g)
    @property
    def brgb(self): return Int4(self.b, self.r, self.g, self.b)
    @property
    def brbr(self): return Int4(self.b, self.r, self.b, self.r)
    @property
    def brbg(self): return Int4(self.b, self.r, self.b, self.g)
    @property
    def brbb(self): return Int4(self.b, self.r, self.b, self.b)
    @property
    def bgrr(self): return Int4(self.b, self.g, self.r, self.r)
    @property
    def bgrg(self): return Int4(self.b, self.g, self.r, self.g)
    @property
    def bgrb(self): return Int4(self.b, self.g, self.r, self.b)
    @property
    def bggr(self): return Int4(self.b, self.g, self.g, self.r)
    @property
    def bggg(self): return Int4(self.b, self.g, self.g, self.g)
    @property
    def bggb(self): return Int4(self.b, self.g, self.g, self.b)
    @property
    def bgbr(self): return Int4(self.b, self.g, self.b, self.r)
    @property
    def bgbg(self): return Int4(self.b, self.g, self.b, self.g)
    @property
    def bgbb(self): return Int4(self.b, self.g, self.b, self.b)
    @property
    def bbrr(self): return Int4(self.b, self.b, self.r, self.r)
    @property
    def bbrg(self): return Int4(self.b, self.b, self.r, self.g)
    @property
    def bbrb(self): return Int4(self.b, self.b, self.r, self.b)
    @property
    def bbgr(self): return Int4(self.b, self.b, self.g, self.r)
    @property
    def bbgg(self): return Int4(self.b, self.b, self.g, self.g)
    @property
    def bbgb(self): return Int4(self.b, self.b, self.g, self.b)
    @property
    def bbbr(self): return Int4(self.b, self.b, self.b, self.r)
    @property
    def bbbg(self): return Int4(self.b, self.b, self.b, self.g)
    @property
    def bbbb(self): return Int4(self.b, self.b, self.b, self.b)

class Int4(np.ndarray):
    """
    Int4 type using numpy.ndarray.
    """
    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                assert len(args[0]) == 4, "list/tuple must have 3 components"
                arr = np.asarray([args[0][0], args[0][1], args[0][2], args[0][3]], dtype=np.int32).view(cls)
            elif isinstance(args[0], np.ndarray):
                assert len(args[0].squeeze().shape) == 1 and args[0].shape[0] == 4, \
                    "numpy array must be sized [C=4] or [C=4, H=1, W=1]"
                arr = np.asarray(args[0].squeeze(), dtype=np.int32).view(cls)
            elif torch.is_tensor(args[0]):
                assert len(args[0].squeeze().size()) == 1 and args[0].size(0) == 4, \
                    "torch tensor must be sized [C=4] or [C=4, H=1, W=1]"
                value = args[0].squeeze().int().cpu()
                arr = np.asarray([value[0].item(), value[1].item(), value[2].item(), value[3].item()], dtype=np.int32).view(cls)
            else:
                value = int(args[0])
                arr = np.asarray([value, value, value, value], dtype=np.int32).view(cls)
        elif len(args) == 4:
            arr = np.asarray(args, dtype=np.int32).view(cls)
        else: 
            raise TypeError("Int4 only accepts 1 or 4 arguments.")
        return arr

    def list(self) -> list:
        """Returns values as Python list"""
        return [self[0], self[1], self[2], self[3]]

    def tuple(self) -> tuple:
        """Returns values as Python tuple"""
        return (self[0], self[1], self[2], self[3])

    @property
    def x(self) -> int:
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value
    @property
    def y(self) -> int:
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value
    @property
    def z(self) -> int:
        return self[2]
    @z.setter
    def z(self, value):
        self[2] = value
    @property
    def w(self) -> int:
        return self[3]
    @w.setter
    def w(self, value):
        self[3] = value
    @property
    def r(self) -> int:
        return self[0]
    @r.setter
    def r(self, value):
        self[0] = value
    @property
    def g(self) -> int:
        return self[1]
    @g.setter
    def g(self, value):
        self[1] = value
    @property
    def b(self) -> int:
        return self[2]
    @b.setter
    def b(self, value):
        self[2] = value
    @property
    def a(self) -> int:
        return self[3]
    @a.setter
    def a(self, value):
        self[3] = value

    @property
    def xx(self): return Int2(self.x, self.x)
    @property
    def xy(self): return Int2(self.x, self.y)
    @property
    def xz(self): return Int2(self.x, self.z)
    @property
    def xw(self): return Int2(self.x, self.w)
    @property
    def yx(self): return Int2(self.y, self.x)
    @property
    def yy(self): return Int2(self.y, self.y)
    @property
    def yz(self): return Int2(self.y, self.z)
    @property
    def yw(self): return Int2(self.y, self.w)
    @property
    def zx(self): return Int2(self.z, self.x)
    @property
    def zy(self): return Int2(self.z, self.y)
    @property
    def zz(self): return Int2(self.z, self.z)
    @property
    def zw(self): return Int2(self.z, self.w)
    @property
    def wx(self): return Int2(self.w, self.x)
    @property
    def wy(self): return Int2(self.w, self.y)
    @property
    def wz(self): return Int2(self.w, self.z)
    @property
    def ww(self): return Int2(self.w, self.w)

    @property
    def rr(self): return Int2(self.r, self.r)
    @property
    def rg(self): return Int2(self.r, self.g)
    @property
    def rb(self): return Int2(self.r, self.b)
    @property
    def ra(self): return Int2(self.r, self.a)
    @property
    def gr(self): return Int2(self.g, self.r)
    @property
    def gg(self): return Int2(self.g, self.g)
    @property
    def gb(self): return Int2(self.g, self.b)
    @property
    def ga(self): return Int2(self.g, self.a)
    @property
    def br(self): return Int2(self.b, self.r)
    @property
    def bg(self): return Int2(self.b, self.g)
    @property
    def bb(self): return Int2(self.b, self.b)
    @property
    def ba(self): return Int2(self.b, self.a)
    @property
    def ar(self): return Int2(self.a, self.r)
    @property
    def ag(self): return Int2(self.a, self.g)
    @property
    def ab(self): return Int2(self.a, self.b)
    @property
    def aa(self): return Int2(self.a, self.a)

    @property
    def rrr(self): return Int3(self.r, self.r, self.r)
    @property
    def rrg(self): return Int3(self.r, self.r, self.g)
    @property
    def rrb(self): return Int3(self.r, self.r, self.b)
    @property
    def rra(self): return Int3(self.r, self.r, self.a)
    @property
    def rgr(self): return Int3(self.r, self.g, self.r)
    @property
    def rgg(self): return Int3(self.r, self.g, self.g)
    @property
    def rgb(self): return Int3(self.r, self.g, self.b)
    @property
    def rga(self): return Int3(self.r, self.g, self.a)
    @property
    def rbr(self): return Int3(self.r, self.b, self.r)
    @property
    def rbg(self): return Int3(self.r, self.b, self.g)
    @property
    def rbb(self): return Int3(self.r, self.b, self.b)
    @property
    def rba(self): return Int3(self.r, self.b, self.a)
    @property
    def rar(self): return Int3(self.r, self.a, self.r)
    @property
    def rag(self): return Int3(self.r, self.a, self.g)
    @property
    def rab(self): return Int3(self.r, self.a, self.b)
    @property
    def raa(self): return Int3(self.r, self.a, self.a)
    @property
    def grr(self): return Int3(self.g, self.r, self.r)
    @property
    def grg(self): return Int3(self.g, self.r, self.g)
    @property
    def grb(self): return Int3(self.g, self.r, self.b)
    @property
    def gra(self): return Int3(self.g, self.r, self.a)
    @property
    def ggr(self): return Int3(self.g, self.g, self.r)
    @property
    def ggg(self): return Int3(self.g, self.g, self.g)
    @property
    def ggb(self): return Int3(self.g, self.g, self.b)
    @property
    def gga(self): return Int3(self.g, self.g, self.a)
    @property
    def gbr(self): return Int3(self.g, self.b, self.r)
    @property
    def gbg(self): return Int3(self.g, self.b, self.g)
    @property
    def gbb(self): return Int3(self.g, self.b, self.b)
    @property
    def gba(self): return Int3(self.g, self.b, self.a)
    @property
    def gar(self): return Int3(self.g, self.a, self.r)
    @property
    def gag(self): return Int3(self.g, self.a, self.g)
    @property
    def gab(self): return Int3(self.g, self.a, self.b)
    @property
    def gaa(self): return Int3(self.g, self.a, self.a)
    @property
    def brr(self): return Int3(self.b, self.r, self.r)
    @property
    def brg(self): return Int3(self.b, self.r, self.g)
    @property
    def brb(self): return Int3(self.b, self.r, self.b)
    @property
    def bra(self): return Int3(self.b, self.r, self.a)
    @property
    def bgr(self): return Int3(self.b, self.g, self.r)
    @property
    def bgg(self): return Int3(self.b, self.g, self.g)
    @property
    def bgb(self): return Int3(self.b, self.g, self.b)
    @property
    def bga(self): return Int3(self.b, self.g, self.a)
    @property
    def bbr(self): return Int3(self.b, self.b, self.r)
    @property
    def bbg(self): return Int3(self.b, self.b, self.g)
    @property
    def bbb(self): return Int3(self.b, self.b, self.b)
    @property
    def bba(self): return Int3(self.b, self.b, self.a)
    @property
    def bar(self): return Int3(self.b, self.a, self.r)
    @property
    def bag(self): return Int3(self.b, self.a, self.g)
    @property
    def bab(self): return Int3(self.b, self.a, self.b)
    @property
    def baa(self): return Int3(self.b, self.a, self.a)
    @property
    def arr(self): return Int3(self.a, self.r, self.r)
    @property
    def arg(self): return Int3(self.a, self.r, self.g)
    @property
    def arb(self): return Int3(self.a, self.r, self.b)
    @property
    def ara(self): return Int3(self.a, self.r, self.a)
    @property
    def agr(self): return Int3(self.a, self.g, self.r)
    @property
    def agg(self): return Int3(self.a, self.g, self.g)
    @property
    def agb(self): return Int3(self.a, self.g, self.b)
    @property
    def aga(self): return Int3(self.a, self.g, self.a)
    @property
    def abr(self): return Int3(self.a, self.b, self.r)
    @property
    def abg(self): return Int3(self.a, self.b, self.g)
    @property
    def abb(self): return Int3(self.a, self.b, self.b)
    @property
    def aba(self): return Int3(self.a, self.b, self.a)
    @property
    def aar(self): return Int3(self.a, self.a, self.r)
    @property
    def aag(self): return Int3(self.a, self.a, self.g)
    @property
    def aab(self): return Int3(self.a, self.a, self.b)
    @property
    def aaa(self): return Int3(self.a, self.a, self.a)

    @property
    def xxx(self): return Int3(self.x, self.x, self.x)
    @property
    def xxy(self): return Int3(self.x, self.x, self.y)
    @property
    def xxz(self): return Int3(self.x, self.x, self.z)
    @property
    def xxw(self): return Int3(self.x, self.x, self.w)
    @property
    def xyx(self): return Int3(self.x, self.y, self.x)
    @property
    def xyy(self): return Int3(self.x, self.y, self.y)
    @property
    def xyz(self): return Int3(self.x, self.y, self.z)
    @property
    def xyw(self): return Int3(self.x, self.y, self.w)
    @property
    def xzx(self): return Int3(self.x, self.z, self.x)
    @property
    def xzy(self): return Int3(self.x, self.z, self.y)
    @property
    def xzz(self): return Int3(self.x, self.z, self.z)
    @property
    def xzw(self): return Int3(self.x, self.z, self.w)
    @property
    def xwx(self): return Int3(self.x, self.w, self.x)
    @property
    def xwy(self): return Int3(self.x, self.w, self.y)
    @property
    def xwz(self): return Int3(self.x, self.w, self.z)
    @property
    def xww(self): return Int3(self.x, self.w, self.w)
    @property
    def yxx(self): return Int3(self.y, self.x, self.x)
    @property
    def yxy(self): return Int3(self.y, self.x, self.y)
    @property
    def yxz(self): return Int3(self.y, self.x, self.z)
    @property
    def yxw(self): return Int3(self.y, self.x, self.w)
    @property
    def yyx(self): return Int3(self.y, self.y, self.x)
    @property
    def yyy(self): return Int3(self.y, self.y, self.y)
    @property
    def yyz(self): return Int3(self.y, self.y, self.z)
    @property
    def yyw(self): return Int3(self.y, self.y, self.w)
    @property
    def yzx(self): return Int3(self.y, self.z, self.x)
    @property
    def yzy(self): return Int3(self.y, self.z, self.y)
    @property
    def yzz(self): return Int3(self.y, self.z, self.z)
    @property
    def yzw(self): return Int3(self.y, self.z, self.w)
    @property
    def ywx(self): return Int3(self.y, self.w, self.x)
    @property
    def ywy(self): return Int3(self.y, self.w, self.y)
    @property
    def ywz(self): return Int3(self.y, self.w, self.z)
    @property
    def yww(self): return Int3(self.y, self.w, self.w)
    @property
    def zxx(self): return Int3(self.z, self.x, self.x)
    @property
    def zxy(self): return Int3(self.z, self.x, self.y)
    @property
    def zxz(self): return Int3(self.z, self.x, self.z)
    @property
    def zxw(self): return Int3(self.z, self.x, self.w)
    @property
    def zyx(self): return Int3(self.z, self.y, self.x)
    @property
    def zyy(self): return Int3(self.z, self.y, self.y)
    @property
    def zyz(self): return Int3(self.z, self.y, self.z)
    @property
    def zyw(self): return Int3(self.z, self.y, self.w)
    @property
    def zzx(self): return Int3(self.z, self.z, self.x)
    @property
    def zzy(self): return Int3(self.z, self.z, self.y)
    @property
    def zzz(self): return Int3(self.z, self.z, self.z)
    @property
    def zzw(self): return Int3(self.z, self.z, self.w)
    @property
    def zwx(self): return Int3(self.z, self.w, self.x)
    @property
    def zwy(self): return Int3(self.z, self.w, self.y)
    @property
    def zwz(self): return Int3(self.z, self.w, self.z)
    @property
    def zww(self): return Int3(self.z, self.w, self.w)
    @property
    def wxx(self): return Int3(self.w, self.x, self.x)
    @property
    def wxy(self): return Int3(self.w, self.x, self.y)
    @property
    def wxz(self): return Int3(self.w, self.x, self.z)
    @property
    def wxw(self): return Int3(self.w, self.x, self.w)
    @property
    def wyx(self): return Int3(self.w, self.y, self.x)
    @property
    def wyy(self): return Int3(self.w, self.y, self.y)
    @property
    def wyz(self): return Int3(self.w, self.y, self.z)
    @property
    def wyw(self): return Int3(self.w, self.y, self.w)
    @property
    def wzx(self): return Int3(self.w, self.z, self.x)
    @property
    def wzy(self): return Int3(self.w, self.z, self.y)
    @property
    def wzz(self): return Int3(self.w, self.z, self.z)
    @property
    def wzw(self): return Int3(self.w, self.z, self.w)
    @property
    def wwx(self): return Int3(self.w, self.w, self.x)
    @property
    def wwy(self): return Int3(self.w, self.w, self.y)
    @property
    def wwz(self): return Int3(self.w, self.w, self.z)
    @property
    def www(self): return Int3(self.w, self.w, self.w)

    @property
    def xxxx(self): return Int4(self.x, self.x, self.x, self.x)
    @property
    def xxxy(self): return Int4(self.x, self.x, self.x, self.y)
    @property
    def xxxz(self): return Int4(self.x, self.x, self.x, self.z)
    @property
    def xxxw(self): return Int4(self.x, self.x, self.x, self.w)
    @property
    def xxyx(self): return Int4(self.x, self.x, self.y, self.x)
    @property
    def xxyy(self): return Int4(self.x, self.x, self.y, self.y)
    @property
    def xxyz(self): return Int4(self.x, self.x, self.y, self.z)
    @property
    def xxyw(self): return Int4(self.x, self.x, self.y, self.w)
    @property
    def xxzx(self): return Int4(self.x, self.x, self.z, self.x)
    @property
    def xxzy(self): return Int4(self.x, self.x, self.z, self.y)
    @property
    def xxzz(self): return Int4(self.x, self.x, self.z, self.z)
    @property
    def xxzw(self): return Int4(self.x, self.x, self.z, self.w)
    @property
    def xxwx(self): return Int4(self.x, self.x, self.w, self.x)
    @property
    def xxwy(self): return Int4(self.x, self.x, self.w, self.y)
    @property
    def xxwz(self): return Int4(self.x, self.x, self.w, self.z)
    @property
    def xxww(self): return Int4(self.x, self.x, self.w, self.w)
    @property
    def xyxx(self): return Int4(self.x, self.y, self.x, self.x)
    @property
    def xyxy(self): return Int4(self.x, self.y, self.x, self.y)
    @property
    def xyxz(self): return Int4(self.x, self.y, self.x, self.z)
    @property
    def xyxw(self): return Int4(self.x, self.y, self.x, self.w)
    @property
    def xyyx(self): return Int4(self.x, self.y, self.y, self.x)
    @property
    def xyyy(self): return Int4(self.x, self.y, self.y, self.y)
    @property
    def xyyz(self): return Int4(self.x, self.y, self.y, self.z)
    @property
    def xyyw(self): return Int4(self.x, self.y, self.y, self.w)
    @property
    def xyzx(self): return Int4(self.x, self.y, self.z, self.x)
    @property
    def xyzy(self): return Int4(self.x, self.y, self.z, self.y)
    @property
    def xyzz(self): return Int4(self.x, self.y, self.z, self.z)
    @property
    def xyzw(self): return self
    @property
    def xywx(self): return Int4(self.x, self.y, self.w, self.x)
    @property
    def xywy(self): return Int4(self.x, self.y, self.w, self.y)
    @property
    def xywz(self): return Int4(self.x, self.y, self.w, self.z)
    @property
    def xyww(self): return Int4(self.x, self.y, self.w, self.w)
    @property
    def xzxx(self): return Int4(self.x, self.z, self.x, self.x)
    @property
    def xzxy(self): return Int4(self.x, self.z, self.x, self.y)
    @property
    def xzxz(self): return Int4(self.x, self.z, self.x, self.z)
    @property
    def xzxw(self): return Int4(self.x, self.z, self.x, self.w)
    @property
    def xzyx(self): return Int4(self.x, self.z, self.y, self.x)
    @property
    def xzyy(self): return Int4(self.x, self.z, self.y, self.y)
    @property
    def xzyz(self): return Int4(self.x, self.z, self.y, self.z)
    @property
    def xzyw(self): return Int4(self.x, self.z, self.y, self.w)
    @property
    def xzzx(self): return Int4(self.x, self.z, self.z, self.x)
    @property
    def xzzy(self): return Int4(self.x, self.z, self.z, self.y)
    @property
    def xzzz(self): return Int4(self.x, self.z, self.z, self.z)
    @property
    def xzzw(self): return Int4(self.x, self.z, self.z, self.w)
    @property
    def xzwx(self): return Int4(self.x, self.z, self.w, self.x)
    @property
    def xzwy(self): return Int4(self.x, self.z, self.w, self.y)
    @property
    def xzwz(self): return Int4(self.x, self.z, self.w, self.z)
    @property
    def xzww(self): return Int4(self.x, self.z, self.w, self.w)
    @property
    def xwxx(self): return Int4(self.x, self.w, self.x, self.x)
    @property
    def xwxy(self): return Int4(self.x, self.w, self.x, self.y)
    @property
    def xwxz(self): return Int4(self.x, self.w, self.x, self.z)
    @property
    def xwxw(self): return Int4(self.x, self.w, self.x, self.w)
    @property
    def xwyx(self): return Int4(self.x, self.w, self.y, self.x)
    @property
    def xwyy(self): return Int4(self.x, self.w, self.y, self.y)
    @property
    def xwyz(self): return Int4(self.x, self.w, self.y, self.z)
    @property
    def xwyw(self): return Int4(self.x, self.w, self.y, self.w)
    @property
    def xwzx(self): return Int4(self.x, self.w, self.z, self.x)
    @property
    def xwzy(self): return Int4(self.x, self.w, self.z, self.y)
    @property
    def xwzz(self): return Int4(self.x, self.w, self.z, self.z)
    @property
    def xwzw(self): return Int4(self.x, self.w, self.z, self.w)
    @property
    def xwwx(self): return Int4(self.x, self.w, self.w, self.x)
    @property
    def xwwy(self): return Int4(self.x, self.w, self.w, self.y)
    @property
    def xwwz(self): return Int4(self.x, self.w, self.w, self.z)
    @property
    def xwww(self): return Int4(self.x, self.w, self.w, self.w)
    @property
    def yxxx(self): return Int4(self.y, self.x, self.x, self.x)
    @property
    def yxxy(self): return Int4(self.y, self.x, self.x, self.y)
    @property
    def yxxz(self): return Int4(self.y, self.x, self.x, self.z)
    @property
    def yxxw(self): return Int4(self.y, self.x, self.x, self.w)
    @property
    def yxyx(self): return Int4(self.y, self.x, self.y, self.x)
    @property
    def yxyy(self): return Int4(self.y, self.x, self.y, self.y)
    @property
    def yxyz(self): return Int4(self.y, self.x, self.y, self.z)
    @property
    def yxyw(self): return Int4(self.y, self.x, self.y, self.w)
    @property
    def yxzx(self): return Int4(self.y, self.x, self.z, self.x)
    @property
    def yxzy(self): return Int4(self.y, self.x, self.z, self.y)
    @property
    def yxzz(self): return Int4(self.y, self.x, self.z, self.z)
    @property
    def yxzw(self): return Int4(self.y, self.x, self.z, self.w)
    @property
    def yxwx(self): return Int4(self.y, self.x, self.w, self.x)
    @property
    def yxwy(self): return Int4(self.y, self.x, self.w, self.y)
    @property
    def yxwz(self): return Int4(self.y, self.x, self.w, self.z)
    @property
    def yxww(self): return Int4(self.y, self.x, self.w, self.w)
    @property
    def yyxx(self): return Int4(self.y, self.y, self.x, self.x)
    @property
    def yyxy(self): return Int4(self.y, self.y, self.x, self.y)
    @property
    def yyxz(self): return Int4(self.y, self.y, self.x, self.z)
    @property
    def yyxw(self): return Int4(self.y, self.y, self.x, self.w)
    @property
    def yyyx(self): return Int4(self.y, self.y, self.y, self.x)
    @property
    def yyyy(self): return Int4(self.y, self.y, self.y, self.y)
    @property
    def yyyz(self): return Int4(self.y, self.y, self.y, self.z)
    @property
    def yyyw(self): return Int4(self.y, self.y, self.y, self.w)
    @property
    def yyzx(self): return Int4(self.y, self.y, self.z, self.x)
    @property
    def yyzy(self): return Int4(self.y, self.y, self.z, self.y)
    @property
    def yyzz(self): return Int4(self.y, self.y, self.z, self.z)
    @property
    def yyzw(self): return Int4(self.y, self.y, self.z, self.w)
    @property
    def yywx(self): return Int4(self.y, self.y, self.w, self.x)
    @property
    def yywy(self): return Int4(self.y, self.y, self.w, self.y)
    @property
    def yywz(self): return Int4(self.y, self.y, self.w, self.z)
    @property
    def yyww(self): return Int4(self.y, self.y, self.w, self.w)
    @property
    def yzxx(self): return Int4(self.y, self.z, self.x, self.x)
    @property
    def yzxy(self): return Int4(self.y, self.z, self.x, self.y)
    @property
    def yzxz(self): return Int4(self.y, self.z, self.x, self.z)
    @property
    def yzxw(self): return Int4(self.y, self.z, self.x, self.w)
    @property
    def yzyx(self): return Int4(self.y, self.z, self.y, self.x)
    @property
    def yzyy(self): return Int4(self.y, self.z, self.y, self.y)
    @property
    def yzyz(self): return Int4(self.y, self.z, self.y, self.z)
    @property
    def yzyw(self): return Int4(self.y, self.z, self.y, self.w)
    @property
    def yzzx(self): return Int4(self.y, self.z, self.z, self.x)
    @property
    def yzzy(self): return Int4(self.y, self.z, self.z, self.y)
    @property
    def yzzz(self): return Int4(self.y, self.z, self.z, self.z)
    @property
    def yzzw(self): return Int4(self.y, self.z, self.z, self.w)
    @property
    def yzwx(self): return Int4(self.y, self.z, self.w, self.x)
    @property
    def yzwy(self): return Int4(self.y, self.z, self.w, self.y)
    @property
    def yzwz(self): return Int4(self.y, self.z, self.w, self.z)
    @property
    def yzww(self): return Int4(self.y, self.z, self.w, self.w)
    @property
    def ywxx(self): return Int4(self.y, self.w, self.x, self.x)
    @property
    def ywxy(self): return Int4(self.y, self.w, self.x, self.y)
    @property
    def ywxz(self): return Int4(self.y, self.w, self.x, self.z)
    @property
    def ywxw(self): return Int4(self.y, self.w, self.x, self.w)
    @property
    def ywyx(self): return Int4(self.y, self.w, self.y, self.x)
    @property
    def ywyy(self): return Int4(self.y, self.w, self.y, self.y)
    @property
    def ywyz(self): return Int4(self.y, self.w, self.y, self.z)
    @property
    def ywyw(self): return Int4(self.y, self.w, self.y, self.w)
    @property
    def ywzx(self): return Int4(self.y, self.w, self.z, self.x)
    @property
    def ywzy(self): return Int4(self.y, self.w, self.z, self.y)
    @property
    def ywzz(self): return Int4(self.y, self.w, self.z, self.z)
    @property
    def ywzw(self): return Int4(self.y, self.w, self.z, self.w)
    @property
    def ywwx(self): return Int4(self.y, self.w, self.w, self.x)
    @property
    def ywwy(self): return Int4(self.y, self.w, self.w, self.y)
    @property
    def ywwz(self): return Int4(self.y, self.w, self.w, self.z)
    @property
    def ywww(self): return Int4(self.y, self.w, self.w, self.w)
    @property
    def zxxx(self): return Int4(self.z, self.x, self.x, self.x)
    @property
    def zxxy(self): return Int4(self.z, self.x, self.x, self.y)
    @property
    def zxxz(self): return Int4(self.z, self.x, self.x, self.z)
    @property
    def zxxw(self): return Int4(self.z, self.x, self.x, self.w)
    @property
    def zxyx(self): return Int4(self.z, self.x, self.y, self.x)
    @property
    def zxyy(self): return Int4(self.z, self.x, self.y, self.y)
    @property
    def zxyz(self): return Int4(self.z, self.x, self.y, self.z)
    @property
    def zxyw(self): return Int4(self.z, self.x, self.y, self.w)
    @property
    def zxzx(self): return Int4(self.z, self.x, self.z, self.x)
    @property
    def zxzy(self): return Int4(self.z, self.x, self.z, self.y)
    @property
    def zxzz(self): return Int4(self.z, self.x, self.z, self.z)
    @property
    def zxzw(self): return Int4(self.z, self.x, self.z, self.w)
    @property
    def zxwx(self): return Int4(self.z, self.x, self.w, self.x)
    @property
    def zxwy(self): return Int4(self.z, self.x, self.w, self.y)
    @property
    def zxwz(self): return Int4(self.z, self.x, self.w, self.z)
    @property
    def zxww(self): return Int4(self.z, self.x, self.w, self.w)
    @property
    def zyxx(self): return Int4(self.z, self.y, self.x, self.x)
    @property
    def zyxy(self): return Int4(self.z, self.y, self.x, self.y)
    @property
    def zyxz(self): return Int4(self.z, self.y, self.x, self.z)
    @property
    def zyxw(self): return Int4(self.z, self.y, self.x, self.w)
    @property
    def zyyx(self): return Int4(self.z, self.y, self.y, self.x)
    @property
    def zyyy(self): return Int4(self.z, self.y, self.y, self.y)
    @property
    def zyyz(self): return Int4(self.z, self.y, self.y, self.z)
    @property
    def zyyw(self): return Int4(self.z, self.y, self.y, self.w)
    @property
    def zyzx(self): return Int4(self.z, self.y, self.z, self.x)
    @property
    def zyzy(self): return Int4(self.z, self.y, self.z, self.y)
    @property
    def zyzz(self): return Int4(self.z, self.y, self.z, self.z)
    @property
    def zyzw(self): return Int4(self.z, self.y, self.z, self.w)
    @property
    def zywx(self): return Int4(self.z, self.y, self.w, self.x)
    @property
    def zywy(self): return Int4(self.z, self.y, self.w, self.y)
    @property
    def zywz(self): return Int4(self.z, self.y, self.w, self.z)
    @property
    def zyww(self): return Int4(self.z, self.y, self.w, self.w)
    @property
    def zzxx(self): return Int4(self.z, self.z, self.x, self.x)
    @property
    def zzxy(self): return Int4(self.z, self.z, self.x, self.y)
    @property
    def zzxz(self): return Int4(self.z, self.z, self.x, self.z)
    @property
    def zzxw(self): return Int4(self.z, self.z, self.x, self.w)
    @property
    def zzyx(self): return Int4(self.z, self.z, self.y, self.x)
    @property
    def zzyy(self): return Int4(self.z, self.z, self.y, self.y)
    @property
    def zzyz(self): return Int4(self.z, self.z, self.y, self.z)
    @property
    def zzyw(self): return Int4(self.z, self.z, self.y, self.w)
    @property
    def zzzx(self): return Int4(self.z, self.z, self.z, self.x)
    @property
    def zzzy(self): return Int4(self.z, self.z, self.z, self.y)
    @property
    def zzzz(self): return Int4(self.z, self.z, self.z, self.z)
    @property
    def zzzw(self): return Int4(self.z, self.z, self.z, self.w)
    @property
    def zzwx(self): return Int4(self.z, self.z, self.w, self.x)
    @property
    def zzwy(self): return Int4(self.z, self.z, self.w, self.y)
    @property
    def zzwz(self): return Int4(self.z, self.z, self.w, self.z)
    @property
    def zzww(self): return Int4(self.z, self.z, self.w, self.w)
    @property
    def zwxx(self): return Int4(self.z, self.w, self.x, self.x)
    @property
    def zwxy(self): return Int4(self.z, self.w, self.x, self.y)
    @property
    def zwxz(self): return Int4(self.z, self.w, self.x, self.z)
    @property
    def zwxw(self): return Int4(self.z, self.w, self.x, self.w)
    @property
    def zwyx(self): return Int4(self.z, self.w, self.y, self.x)
    @property
    def zwyy(self): return Int4(self.z, self.w, self.y, self.y)
    @property
    def zwyz(self): return Int4(self.z, self.w, self.y, self.z)
    @property
    def zwyw(self): return Int4(self.z, self.w, self.y, self.w)
    @property
    def zwzx(self): return Int4(self.z, self.w, self.z, self.x)
    @property
    def zwzy(self): return Int4(self.z, self.w, self.z, self.y)
    @property
    def zwzz(self): return Int4(self.z, self.w, self.z, self.z)
    @property
    def zwzw(self): return Int4(self.z, self.w, self.z, self.w)
    @property
    def zwwx(self): return Int4(self.z, self.w, self.w, self.x)
    @property
    def zwwy(self): return Int4(self.z, self.w, self.w, self.y)
    @property
    def zwwz(self): return Int4(self.z, self.w, self.w, self.z)
    @property
    def zwww(self): return Int4(self.z, self.w, self.w, self.w)
    @property
    def wxxx(self): return Int4(self.w, self.x, self.x, self.x)
    @property
    def wxxy(self): return Int4(self.w, self.x, self.x, self.y)
    @property
    def wxxz(self): return Int4(self.w, self.x, self.x, self.z)
    @property
    def wxxw(self): return Int4(self.w, self.x, self.x, self.w)
    @property
    def wxyx(self): return Int4(self.w, self.x, self.y, self.x)
    @property
    def wxyy(self): return Int4(self.w, self.x, self.y, self.y)
    @property
    def wxyz(self): return Int4(self.w, self.x, self.y, self.z)
    @property
    def wxyw(self): return Int4(self.w, self.x, self.y, self.w)
    @property
    def wxzx(self): return Int4(self.w, self.x, self.z, self.x)
    @property
    def wxzy(self): return Int4(self.w, self.x, self.z, self.y)
    @property
    def wxzz(self): return Int4(self.w, self.x, self.z, self.z)
    @property
    def wxzw(self): return Int4(self.w, self.x, self.z, self.w)
    @property
    def wxwx(self): return Int4(self.w, self.x, self.w, self.x)
    @property
    def wxwy(self): return Int4(self.w, self.x, self.w, self.y)
    @property
    def wxwz(self): return Int4(self.w, self.x, self.w, self.z)
    @property
    def wxww(self): return Int4(self.w, self.x, self.w, self.w)
    @property
    def wyxx(self): return Int4(self.w, self.y, self.x, self.x)
    @property
    def wyxy(self): return Int4(self.w, self.y, self.x, self.y)
    @property
    def wyxz(self): return Int4(self.w, self.y, self.x, self.z)
    @property
    def wyxw(self): return Int4(self.w, self.y, self.x, self.w)
    @property
    def wyyx(self): return Int4(self.w, self.y, self.y, self.x)
    @property
    def wyyy(self): return Int4(self.w, self.y, self.y, self.y)
    @property
    def wyyz(self): return Int4(self.w, self.y, self.y, self.z)
    @property
    def wyyw(self): return Int4(self.w, self.y, self.y, self.w)
    @property
    def wyzx(self): return Int4(self.w, self.y, self.z, self.x)
    @property
    def wyzy(self): return Int4(self.w, self.y, self.z, self.y)
    @property
    def wyzz(self): return Int4(self.w, self.y, self.z, self.z)
    @property
    def wyzw(self): return Int4(self.w, self.y, self.z, self.w)
    @property
    def wywx(self): return Int4(self.w, self.y, self.w, self.x)
    @property
    def wywy(self): return Int4(self.w, self.y, self.w, self.y)
    @property
    def wywz(self): return Int4(self.w, self.y, self.w, self.z)
    @property
    def wyww(self): return Int4(self.w, self.y, self.w, self.w)
    @property
    def wzxx(self): return Int4(self.w, self.z, self.x, self.x)
    @property
    def wzxy(self): return Int4(self.w, self.z, self.x, self.y)
    @property
    def wzxz(self): return Int4(self.w, self.z, self.x, self.z)
    @property
    def wzxw(self): return Int4(self.w, self.z, self.x, self.w)
    @property
    def wzyx(self): return Int4(self.w, self.z, self.y, self.x)
    @property
    def wzyy(self): return Int4(self.w, self.z, self.y, self.y)
    @property
    def wzyz(self): return Int4(self.w, self.z, self.y, self.z)
    @property
    def wzyw(self): return Int4(self.w, self.z, self.y, self.w)
    @property
    def wzzx(self): return Int4(self.w, self.z, self.z, self.x)
    @property
    def wzzy(self): return Int4(self.w, self.z, self.z, self.y)
    @property
    def wzzz(self): return Int4(self.w, self.z, self.z, self.z)
    @property
    def wzzw(self): return Int4(self.w, self.z, self.z, self.w)
    @property
    def wzwx(self): return Int4(self.w, self.z, self.w, self.x)
    @property
    def wzwy(self): return Int4(self.w, self.z, self.w, self.y)
    @property
    def wzwz(self): return Int4(self.w, self.z, self.w, self.z)
    @property
    def wzww(self): return Int4(self.w, self.z, self.w, self.w)
    @property
    def wwxx(self): return Int4(self.w, self.w, self.x, self.x)
    @property
    def wwxy(self): return Int4(self.w, self.w, self.x, self.y)
    @property
    def wwxz(self): return Int4(self.w, self.w, self.x, self.z)
    @property
    def wwxw(self): return Int4(self.w, self.w, self.x, self.w)
    @property
    def wwyx(self): return Int4(self.w, self.w, self.y, self.x)
    @property
    def wwyy(self): return Int4(self.w, self.w, self.y, self.y)
    @property
    def wwyz(self): return Int4(self.w, self.w, self.y, self.z)
    @property
    def wwyw(self): return Int4(self.w, self.w, self.y, self.w)
    @property
    def wwzx(self): return Int4(self.w, self.w, self.z, self.x)
    @property
    def wwzy(self): return Int4(self.w, self.w, self.z, self.y)
    @property
    def wwzz(self): return Int4(self.w, self.w, self.z, self.z)
    @property
    def wwzw(self): return Int4(self.w, self.w, self.z, self.w)
    @property
    def wwwx(self): return Int4(self.w, self.w, self.w, self.x)
    @property
    def wwwy(self): return Int4(self.w, self.w, self.w, self.y)
    @property
    def wwwz(self): return Int4(self.w, self.w, self.w, self.z)
    @property
    def wwww(self): return Int4(self.w, self.w, self.w, self.w)

    @property
    def rrrr(self): return Int4(self.r, self.r, self.r, self.r)
    @property
    def rrrg(self): return Int4(self.r, self.r, self.r, self.g)
    @property
    def rrrb(self): return Int4(self.r, self.r, self.r, self.b)
    @property
    def rrra(self): return Int4(self.r, self.r, self.r, self.a)
    @property
    def rrgr(self): return Int4(self.r, self.r, self.g, self.r)
    @property
    def rrgg(self): return Int4(self.r, self.r, self.g, self.g)
    @property
    def rrgb(self): return Int4(self.r, self.r, self.g, self.b)
    @property
    def rrga(self): return Int4(self.r, self.r, self.g, self.a)
    @property
    def rrbr(self): return Int4(self.r, self.r, self.b, self.r)
    @property
    def rrbg(self): return Int4(self.r, self.r, self.b, self.g)
    @property
    def rrbb(self): return Int4(self.r, self.r, self.b, self.b)
    @property
    def rrba(self): return Int4(self.r, self.r, self.b, self.a)
    @property
    def rrar(self): return Int4(self.r, self.r, self.a, self.r)
    @property
    def rrag(self): return Int4(self.r, self.r, self.a, self.g)
    @property
    def rrab(self): return Int4(self.r, self.r, self.a, self.b)
    @property
    def rraa(self): return Int4(self.r, self.r, self.a, self.a)
    @property
    def rgrr(self): return Int4(self.r, self.g, self.r, self.r)
    @property
    def rgrg(self): return Int4(self.r, self.g, self.r, self.g)
    @property
    def rgrb(self): return Int4(self.r, self.g, self.r, self.b)
    @property
    def rgra(self): return Int4(self.r, self.g, self.r, self.a)
    @property
    def rggr(self): return Int4(self.r, self.g, self.g, self.r)
    @property
    def rggg(self): return Int4(self.r, self.g, self.g, self.g)
    @property
    def rggb(self): return Int4(self.r, self.g, self.g, self.b)
    @property
    def rgga(self): return Int4(self.r, self.g, self.g, self.a)
    @property
    def rgbr(self): return Int4(self.r, self.g, self.b, self.r)
    @property
    def rgbg(self): return Int4(self.r, self.g, self.b, self.g)
    @property
    def rgbb(self): return Int4(self.r, self.g, self.b, self.b)
    @property
    def rgba(self): return self
    @property
    def rgar(self): return Int4(self.r, self.g, self.a, self.r)
    @property
    def rgag(self): return Int4(self.r, self.g, self.a, self.g)
    @property
    def rgab(self): return Int4(self.r, self.g, self.a, self.b)
    @property
    def rgaa(self): return Int4(self.r, self.g, self.a, self.a)
    @property
    def rbrr(self): return Int4(self.r, self.b, self.r, self.r)
    @property
    def rbrg(self): return Int4(self.r, self.b, self.r, self.g)
    @property
    def rbrb(self): return Int4(self.r, self.b, self.r, self.b)
    @property
    def rbra(self): return Int4(self.r, self.b, self.r, self.a)
    @property
    def rbgr(self): return Int4(self.r, self.b, self.g, self.r)
    @property
    def rbgg(self): return Int4(self.r, self.b, self.g, self.g)
    @property
    def rbgb(self): return Int4(self.r, self.b, self.g, self.b)
    @property
    def rbga(self): return Int4(self.r, self.b, self.g, self.a)
    @property
    def rbbr(self): return Int4(self.r, self.b, self.b, self.r)
    @property
    def rbbg(self): return Int4(self.r, self.b, self.b, self.g)
    @property
    def rbbb(self): return Int4(self.r, self.b, self.b, self.b)
    @property
    def rbba(self): return Int4(self.r, self.b, self.b, self.a)
    @property
    def rbar(self): return Int4(self.r, self.b, self.a, self.r)
    @property
    def rbag(self): return Int4(self.r, self.b, self.a, self.g)
    @property
    def rbab(self): return Int4(self.r, self.b, self.a, self.b)
    @property
    def rbaa(self): return Int4(self.r, self.b, self.a, self.a)
    @property
    def rarr(self): return Int4(self.r, self.a, self.r, self.r)
    @property
    def rarg(self): return Int4(self.r, self.a, self.r, self.g)
    @property
    def rarb(self): return Int4(self.r, self.a, self.r, self.b)
    @property
    def rara(self): return Int4(self.r, self.a, self.r, self.a)
    @property
    def ragr(self): return Int4(self.r, self.a, self.g, self.r)
    @property
    def ragg(self): return Int4(self.r, self.a, self.g, self.g)
    @property
    def ragb(self): return Int4(self.r, self.a, self.g, self.b)
    @property
    def raga(self): return Int4(self.r, self.a, self.g, self.a)
    @property
    def rabr(self): return Int4(self.r, self.a, self.b, self.r)
    @property
    def rabg(self): return Int4(self.r, self.a, self.b, self.g)
    @property
    def rabb(self): return Int4(self.r, self.a, self.b, self.b)
    @property
    def raba(self): return Int4(self.r, self.a, self.b, self.a)
    @property
    def raar(self): return Int4(self.r, self.a, self.a, self.r)
    @property
    def raag(self): return Int4(self.r, self.a, self.a, self.g)
    @property
    def raab(self): return Int4(self.r, self.a, self.a, self.b)
    @property
    def raaa(self): return Int4(self.r, self.a, self.a, self.a)
    @property
    def grrr(self): return Int4(self.g, self.r, self.r, self.r)
    @property
    def grrg(self): return Int4(self.g, self.r, self.r, self.g)
    @property
    def grrb(self): return Int4(self.g, self.r, self.r, self.b)
    @property
    def grra(self): return Int4(self.g, self.r, self.r, self.a)
    @property
    def grgr(self): return Int4(self.g, self.r, self.g, self.r)
    @property
    def grgg(self): return Int4(self.g, self.r, self.g, self.g)
    @property
    def grgb(self): return Int4(self.g, self.r, self.g, self.b)
    @property
    def grga(self): return Int4(self.g, self.r, self.g, self.a)
    @property
    def grbr(self): return Int4(self.g, self.r, self.b, self.r)
    @property
    def grbg(self): return Int4(self.g, self.r, self.b, self.g)
    @property
    def grbb(self): return Int4(self.g, self.r, self.b, self.b)
    @property
    def grba(self): return Int4(self.g, self.r, self.b, self.a)
    @property
    def grar(self): return Int4(self.g, self.r, self.a, self.r)
    @property
    def grag(self): return Int4(self.g, self.r, self.a, self.g)
    @property
    def grab(self): return Int4(self.g, self.r, self.a, self.b)
    @property
    def graa(self): return Int4(self.g, self.r, self.a, self.a)
    @property
    def ggrr(self): return Int4(self.g, self.g, self.r, self.r)
    @property
    def ggrg(self): return Int4(self.g, self.g, self.r, self.g)
    @property
    def ggrb(self): return Int4(self.g, self.g, self.r, self.b)
    @property
    def ggra(self): return Int4(self.g, self.g, self.r, self.a)
    @property
    def gggr(self): return Int4(self.g, self.g, self.g, self.r)
    @property
    def gggg(self): return Int4(self.g, self.g, self.g, self.g)
    @property
    def gggb(self): return Int4(self.g, self.g, self.g, self.b)
    @property
    def ggga(self): return Int4(self.g, self.g, self.g, self.a)
    @property
    def ggbr(self): return Int4(self.g, self.g, self.b, self.r)
    @property
    def ggbg(self): return Int4(self.g, self.g, self.b, self.g)
    @property
    def ggbb(self): return Int4(self.g, self.g, self.b, self.b)
    @property
    def ggba(self): return Int4(self.g, self.g, self.b, self.a)
    @property
    def ggar(self): return Int4(self.g, self.g, self.a, self.r)
    @property
    def ggag(self): return Int4(self.g, self.g, self.a, self.g)
    @property
    def ggab(self): return Int4(self.g, self.g, self.a, self.b)
    @property
    def ggaa(self): return Int4(self.g, self.g, self.a, self.a)
    @property
    def gbrr(self): return Int4(self.g, self.b, self.r, self.r)
    @property
    def gbrg(self): return Int4(self.g, self.b, self.r, self.g)
    @property
    def gbrb(self): return Int4(self.g, self.b, self.r, self.b)
    @property
    def gbra(self): return Int4(self.g, self.b, self.r, self.a)
    @property
    def gbgr(self): return Int4(self.g, self.b, self.g, self.r)
    @property
    def gbgg(self): return Int4(self.g, self.b, self.g, self.g)
    @property
    def gbgb(self): return Int4(self.g, self.b, self.g, self.b)
    @property
    def gbga(self): return Int4(self.g, self.b, self.g, self.a)
    @property
    def gbbr(self): return Int4(self.g, self.b, self.b, self.r)
    @property
    def gbbg(self): return Int4(self.g, self.b, self.b, self.g)
    @property
    def gbbb(self): return Int4(self.g, self.b, self.b, self.b)
    @property
    def gbba(self): return Int4(self.g, self.b, self.b, self.a)
    @property
    def gbar(self): return Int4(self.g, self.b, self.a, self.r)
    @property
    def gbag(self): return Int4(self.g, self.b, self.a, self.g)
    @property
    def gbab(self): return Int4(self.g, self.b, self.a, self.b)
    @property
    def gbaa(self): return Int4(self.g, self.b, self.a, self.a)
    @property
    def garr(self): return Int4(self.g, self.a, self.r, self.r)
    @property
    def garg(self): return Int4(self.g, self.a, self.r, self.g)
    @property
    def garb(self): return Int4(self.g, self.a, self.r, self.b)
    @property
    def gara(self): return Int4(self.g, self.a, self.r, self.a)
    @property
    def gagr(self): return Int4(self.g, self.a, self.g, self.r)
    @property
    def gagg(self): return Int4(self.g, self.a, self.g, self.g)
    @property
    def gagb(self): return Int4(self.g, self.a, self.g, self.b)
    @property
    def gaga(self): return Int4(self.g, self.a, self.g, self.a)
    @property
    def gabr(self): return Int4(self.g, self.a, self.b, self.r)
    @property
    def gabg(self): return Int4(self.g, self.a, self.b, self.g)
    @property
    def gabb(self): return Int4(self.g, self.a, self.b, self.b)
    @property
    def gaba(self): return Int4(self.g, self.a, self.b, self.a)
    @property
    def gaar(self): return Int4(self.g, self.a, self.a, self.r)
    @property
    def gaag(self): return Int4(self.g, self.a, self.a, self.g)
    @property
    def gaab(self): return Int4(self.g, self.a, self.a, self.b)
    @property
    def gaaa(self): return Int4(self.g, self.a, self.a, self.a)
    @property
    def brrr(self): return Int4(self.b, self.r, self.r, self.r)
    @property
    def brrg(self): return Int4(self.b, self.r, self.r, self.g)
    @property
    def brrb(self): return Int4(self.b, self.r, self.r, self.b)
    @property
    def brra(self): return Int4(self.b, self.r, self.r, self.a)
    @property
    def brgr(self): return Int4(self.b, self.r, self.g, self.r)
    @property
    def brgg(self): return Int4(self.b, self.r, self.g, self.g)
    @property
    def brgb(self): return Int4(self.b, self.r, self.g, self.b)
    @property
    def brga(self): return Int4(self.b, self.r, self.g, self.a)
    @property
    def brbr(self): return Int4(self.b, self.r, self.b, self.r)
    @property
    def brbg(self): return Int4(self.b, self.r, self.b, self.g)
    @property
    def brbb(self): return Int4(self.b, self.r, self.b, self.b)
    @property
    def brba(self): return Int4(self.b, self.r, self.b, self.a)
    @property
    def brar(self): return Int4(self.b, self.r, self.a, self.r)
    @property
    def brag(self): return Int4(self.b, self.r, self.a, self.g)
    @property
    def brab(self): return Int4(self.b, self.r, self.a, self.b)
    @property
    def braa(self): return Int4(self.b, self.r, self.a, self.a)
    @property
    def bgrr(self): return Int4(self.b, self.g, self.r, self.r)
    @property
    def bgrg(self): return Int4(self.b, self.g, self.r, self.g)
    @property
    def bgrb(self): return Int4(self.b, self.g, self.r, self.b)
    @property
    def bgra(self): return Int4(self.b, self.g, self.r, self.a)
    @property
    def bggr(self): return Int4(self.b, self.g, self.g, self.r)
    @property
    def bggg(self): return Int4(self.b, self.g, self.g, self.g)
    @property
    def bggb(self): return Int4(self.b, self.g, self.g, self.b)
    @property
    def bgga(self): return Int4(self.b, self.g, self.g, self.a)
    @property
    def bgbr(self): return Int4(self.b, self.g, self.b, self.r)
    @property
    def bgbg(self): return Int4(self.b, self.g, self.b, self.g)
    @property
    def bgbb(self): return Int4(self.b, self.g, self.b, self.b)
    @property
    def bgba(self): return Int4(self.b, self.g, self.b, self.a)
    @property
    def bgar(self): return Int4(self.b, self.g, self.a, self.r)
    @property
    def bgag(self): return Int4(self.b, self.g, self.a, self.g)
    @property
    def bgab(self): return Int4(self.b, self.g, self.a, self.b)
    @property
    def bgaa(self): return Int4(self.b, self.g, self.a, self.a)
    @property
    def bbrr(self): return Int4(self.b, self.b, self.r, self.r)
    @property
    def bbrg(self): return Int4(self.b, self.b, self.r, self.g)
    @property
    def bbrb(self): return Int4(self.b, self.b, self.r, self.b)
    @property
    def bbra(self): return Int4(self.b, self.b, self.r, self.a)
    @property
    def bbgr(self): return Int4(self.b, self.b, self.g, self.r)
    @property
    def bbgg(self): return Int4(self.b, self.b, self.g, self.g)
    @property
    def bbgb(self): return Int4(self.b, self.b, self.g, self.b)
    @property
    def bbga(self): return Int4(self.b, self.b, self.g, self.a)
    @property
    def bbbr(self): return Int4(self.b, self.b, self.b, self.r)
    @property
    def bbbg(self): return Int4(self.b, self.b, self.b, self.g)
    @property
    def bbbb(self): return Int4(self.b, self.b, self.b, self.b)
    @property
    def bbba(self): return Int4(self.b, self.b, self.b, self.a)
    @property
    def bbar(self): return Int4(self.b, self.b, self.a, self.r)
    @property
    def bbag(self): return Int4(self.b, self.b, self.a, self.g)
    @property
    def bbab(self): return Int4(self.b, self.b, self.a, self.b)
    @property
    def bbaa(self): return Int4(self.b, self.b, self.a, self.a)
    @property
    def barr(self): return Int4(self.b, self.a, self.r, self.r)
    @property
    def barg(self): return Int4(self.b, self.a, self.r, self.g)
    @property
    def barb(self): return Int4(self.b, self.a, self.r, self.b)
    @property
    def bara(self): return Int4(self.b, self.a, self.r, self.a)
    @property
    def bagr(self): return Int4(self.b, self.a, self.g, self.r)
    @property
    def bagg(self): return Int4(self.b, self.a, self.g, self.g)
    @property
    def bagb(self): return Int4(self.b, self.a, self.g, self.b)
    @property
    def baga(self): return Int4(self.b, self.a, self.g, self.a)
    @property
    def babr(self): return Int4(self.b, self.a, self.b, self.r)
    @property
    def babg(self): return Int4(self.b, self.a, self.b, self.g)
    @property
    def babb(self): return Int4(self.b, self.a, self.b, self.b)
    @property
    def baba(self): return Int4(self.b, self.a, self.b, self.a)
    @property
    def baar(self): return Int4(self.b, self.a, self.a, self.r)
    @property
    def baag(self): return Int4(self.b, self.a, self.a, self.g)
    @property
    def baab(self): return Int4(self.b, self.a, self.a, self.b)
    @property
    def baaa(self): return Int4(self.b, self.a, self.a, self.a)
    @property
    def arrr(self): return Int4(self.a, self.r, self.r, self.r)
    @property
    def arrg(self): return Int4(self.a, self.r, self.r, self.g)
    @property
    def arrb(self): return Int4(self.a, self.r, self.r, self.b)
    @property
    def arra(self): return Int4(self.a, self.r, self.r, self.a)
    @property
    def argr(self): return Int4(self.a, self.r, self.g, self.r)
    @property
    def argg(self): return Int4(self.a, self.r, self.g, self.g)
    @property
    def argb(self): return Int4(self.a, self.r, self.g, self.b)
    @property
    def arga(self): return Int4(self.a, self.r, self.g, self.a)
    @property
    def arbr(self): return Int4(self.a, self.r, self.b, self.r)
    @property
    def arbg(self): return Int4(self.a, self.r, self.b, self.g)
    @property
    def arbb(self): return Int4(self.a, self.r, self.b, self.b)
    @property
    def arba(self): return Int4(self.a, self.r, self.b, self.a)
    @property
    def arar(self): return Int4(self.a, self.r, self.a, self.r)
    @property
    def arag(self): return Int4(self.a, self.r, self.a, self.g)
    @property
    def arab(self): return Int4(self.a, self.r, self.a, self.b)
    @property
    def araa(self): return Int4(self.a, self.r, self.a, self.a)
    @property
    def agrr(self): return Int4(self.a, self.g, self.r, self.r)
    @property
    def agrg(self): return Int4(self.a, self.g, self.r, self.g)
    @property
    def agrb(self): return Int4(self.a, self.g, self.r, self.b)
    @property
    def agra(self): return Int4(self.a, self.g, self.r, self.a)
    @property
    def aggr(self): return Int4(self.a, self.g, self.g, self.r)
    @property
    def aggg(self): return Int4(self.a, self.g, self.g, self.g)
    @property
    def aggb(self): return Int4(self.a, self.g, self.g, self.b)
    @property
    def agga(self): return Int4(self.a, self.g, self.g, self.a)
    @property
    def agbr(self): return Int4(self.a, self.g, self.b, self.r)
    @property
    def agbg(self): return Int4(self.a, self.g, self.b, self.g)
    @property
    def agbb(self): return Int4(self.a, self.g, self.b, self.b)
    @property
    def agba(self): return Int4(self.a, self.g, self.b, self.a)
    @property
    def agar(self): return Int4(self.a, self.g, self.a, self.r)
    @property
    def agag(self): return Int4(self.a, self.g, self.a, self.g)
    @property
    def agab(self): return Int4(self.a, self.g, self.a, self.b)
    @property
    def agaa(self): return Int4(self.a, self.g, self.a, self.a)
    @property
    def abrr(self): return Int4(self.a, self.b, self.r, self.r)
    @property
    def abrg(self): return Int4(self.a, self.b, self.r, self.g)
    @property
    def abrb(self): return Int4(self.a, self.b, self.r, self.b)
    @property
    def abra(self): return Int4(self.a, self.b, self.r, self.a)
    @property
    def abgr(self): return Int4(self.a, self.b, self.g, self.r)
    @property
    def abgg(self): return Int4(self.a, self.b, self.g, self.g)
    @property
    def abgb(self): return Int4(self.a, self.b, self.g, self.b)
    @property
    def abga(self): return Int4(self.a, self.b, self.g, self.a)
    @property
    def abbr(self): return Int4(self.a, self.b, self.b, self.r)
    @property
    def abbg(self): return Int4(self.a, self.b, self.b, self.g)
    @property
    def abbb(self): return Int4(self.a, self.b, self.b, self.b)
    @property
    def abba(self): return Int4(self.a, self.b, self.b, self.a)
    @property
    def abar(self): return Int4(self.a, self.b, self.a, self.r)
    @property
    def abag(self): return Int4(self.a, self.b, self.a, self.g)
    @property
    def abab(self): return Int4(self.a, self.b, self.a, self.b)
    @property
    def abaa(self): return Int4(self.a, self.b, self.a, self.a)
    @property
    def aarr(self): return Int4(self.a, self.a, self.r, self.r)
    @property
    def aarg(self): return Int4(self.a, self.a, self.r, self.g)
    @property
    def aarb(self): return Int4(self.a, self.a, self.r, self.b)
    @property
    def aara(self): return Int4(self.a, self.a, self.r, self.a)
    @property
    def aagr(self): return Int4(self.a, self.a, self.g, self.r)
    @property
    def aagg(self): return Int4(self.a, self.a, self.g, self.g)
    @property
    def aagb(self): return Int4(self.a, self.a, self.g, self.b)
    @property
    def aaga(self): return Int4(self.a, self.a, self.g, self.a)
    @property
    def aabr(self): return Int4(self.a, self.a, self.b, self.r)
    @property
    def aabg(self): return Int4(self.a, self.a, self.b, self.g)
    @property
    def aabb(self): return Int4(self.a, self.a, self.b, self.b)
    @property
    def aaba(self): return Int4(self.a, self.a, self.b, self.a)
    @property
    def aaar(self): return Int4(self.a, self.a, self.a, self.r)
    @property
    def aaag(self): return Int4(self.a, self.a, self.a, self.g)
    @property
    def aaab(self): return Int4(self.a, self.a, self.a, self.b)
    @property
    def aaaa(self): return Int4(self.a, self.a, self.a, self.a)