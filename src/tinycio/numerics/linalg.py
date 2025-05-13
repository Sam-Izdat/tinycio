import torch

from .util_pt import PTUtil
from .util_py import PyUtil

# NOTE: All color-specific code should stay out of this base module

def lerp(a, b, w):
    """
    Linearly interpolate two values.

    :param a: first value or list to interpolate
    :type a: float | list | numpy.ndarray | torch.Tensor
    :param b: second value or list to interpolate 
    :type b: float | list | numpy.ndarray | torch.Tensor
    :param float w: 0-1 weight ratio of first to second value 
    :type w: float | list | numpy.ndarray | torch.Tensor
    :return: linearly interpolated value(s)
    :rtype: float | list | numpy.ndarray | torch.Tensor
    """
    if torch.is_tensor(a) and torch.is_tensor(b): return PTUtil.lerp(a, b, w)
    else: return PyUtil.lerp(a, b, w)
    
def saturate(x):
    """
    Saturate x, such that x is clamped to range [0, 1].

    :param x: input value
    :type x: float | list | numpy.ndarray | torch.Tensor
    :return: saturated input value(s)
    :rtype: float | list | numpy.ndarray | torch.Tensor
    """
    if torch.is_tensor(x): return PTUtil.saturate(x)
    else: return PyUtil.saturate(x)

def sign(x):
    """
    Sign of x as 1, -1 or 0. Returns:

    * 0 if/where x == 0
    * 1 if/where x > 0
    * -1 if/where x < 0

    :param x: input value
    :type x: float | list | numpy.ndarray | torch.Tensor
    :return: sign(s) of input values
    :rtype: float | list | numpy.ndarray | torch.Tensor
    """
    if torch.is_tensor(x): return PTUtil.sign(x)
    else: return PyUtil.sign(x)

def normalize(v):
    """
    Normalize vector v to a unit vector.

    :param v: input vector
    :type v: numpy.ndarray | torch.Tensor
    :return: normalized vector
    :rtype: numpy.ndarray | torch.Tensor
    """
    if torch.is_tensor(v): return PTUtil.normalize(v)
    else: return PyUtil.normalize(v)

def reflect(n, l):
    """
    Reflect vector l over n.

    :param l: input "light" vector
    :type l: numpy.ndarray or torch.Tensor
    :param n: input normal vector
    :type n: numpy.ndarray | torch.Tensor
    :return: reflected vector
    :rtype: numpy.ndarray | torch.Tensor
    """
    if torch.is_tensor(m) and torch.is_tensor(l): return PTUtil.reflect(n, l)
    else: return PyUtil.reflect(n, l)

def matmul(mat:torch.Tensor, im:torch.Tensor):
    """
    Multiply 3x3 tensor matrix and image tensor.

    :param mat: 3x3 matrix for multiplication.
    :type mat: torch.Tensor
    :param im: Input image tensor of shape (C, H, W).
    :type im: torch.Tensor
    :return: Result of the matrix multiplication, with the same shape as the input image.
    :rtype: torch.Tensor
    """
    # NOTE: Internal - leaving this clutter undocumented intentionally
    C, H, W = im.size()
    out = im.permute(1, 2, 0).reshape(-1, 1, C)
    mat = mat.to(im.device).unsqueeze(0).expand(out.size(0), -1, -1)
    out = torch.bmm(out, mat.transpose(1, 2))
    return out.permute(2, 0, 1).view(C, H, W)

def matmul_tl(im:torch.Tensor, mat:list):
    """
    Multiply image tensor by a 3x3 matrix as Python list.

    :param im: Input image tensor of shape (C, H, W).
    :type im: torch.Tensor
    :param mat: 3x3 matrix for multiplication.
    :type mat: List[List[float]]
    :return: Result of the matrix multiplication, with the same shape as the input image.
    :rtype: torch.Tensor
    """
    # NOTE: Internal - leaving this clutter undocumented intentionally
    C, H, W = im.size()
    out = im.clone().permute(1, 2, 0).reshape(-1, 1, C)
    mat = torch.tensor(mat).unsqueeze(0).repeat(out.size(0), 1, 1).to(im.device)
    out = torch.bmm(out, mat.transpose(1, 2)).permute(2,0,1).view(C, H, W)
    return out