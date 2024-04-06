from __future__ import annotations
import typing
from typing import Union

import torch
import numpy as np

from ..numerics import Float2, Float3

def srgb_luminance(im_srgb:Union[torch.Tensor, ColorImage]) -> torch.Tensor:
    """
    Return relative luminance of linear sRGB image.

    :param im_srgb: [C=3, H, W] color image tensor in sRGB color space
    :type im_srgb: torch.Tensor | ColorImage
    :return: [C=1, H, W] image tensor
    """
    lum_r, lum_g, lum_b = 0.2126, 0.7152, 0.0722
    return lum_r * im_srgb[0:1,...] + lum_g * im_srgb[1:2,...] + lum_b * im_srgb[2:3,...]

def apply_gamma(im:Union[torch.Tensor, ColorImage], gamma:float) -> torch.Tensor:
    """
    Apply arbitrary gamma correction.

    :param im: Image tensor
    :type im: torch.Tensor | ColorImage
    :param gamma: Gamma correction (should be in the range [0.1, 10.0])
    :return: Gamma-corrected image tensor
    """
    if gamma == 1.: return im
    assert 0.1 <= gamma <= 10.0, "gamma value should be in range [0.1, 10.0]"
    return torch.pow(im, gamma)

def apply_hue_oklab(im_oklab:Union[torch.Tensor, ColorImage], hue_delta:float) -> torch.Tensor:
    """
    Manually shift hue of an image by a -1 to +1 delta value.

    :param im_oklab: Image tensor in OKLAB color space
    :type im_oklab: torch.Tensor | ColorImage
    :param hue_delta: Hue shift value in the range [-1., 1.]
    :return: Image tensor in OKLAB color space with adjusted hue
    """
    assert -1. <= hue_delta <= 1., "hue_delta value should be in range [-1., 1.]"
    L, a, b = im_oklab[0:1], im_oklab[1:2], im_oklab[2:3]

    hue_delta = ((hue_delta * 0.5) % 1.) * 2. * torch.pi

    # Calculate angle and magnitude in the a-b plane
    angle = torch.atan2(b, a)
    magnitude = torch.sqrt(a**2 + b**2)

    # Apply hue correction
    angle += hue_delta

    # Convert back to Cartesian coordinates
    a_corrected = magnitude * torch.cos(angle)
    b_corrected = magnitude * torch.sin(angle)

    corrected = torch.cat([L, a_corrected, b_corrected], dim=0)
    return corrected

def col_hsv_to_rgb(hsv:Union[Float3, Color]) -> Float3:
    """
    Convert HSV color to RGB.

    :param hsv: HSV color
    :type hsv: Float3 | Color
    """
    h, s, v = hsv.x, hsv.y, hsv.z
    i = np.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]
    return Float3(r, g, b)

def col_rgb_to_hsv(rgb:Union[Float3, Color]) -> Float3:
    """
    Convert RGB color to HSV.

    :param rgb: RGB color
    :type rgb: Float3 | Color
    """
    r, g, b = rgb.r, rgb.g, rgb.b
    high = np.max([r, g, b])
    low = np.min([r, g, b])
    h, s, v = high, high, high
    d = high - low
    s = 0 if high == 0 else d/high
    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6
    return Float3(h, s, v)

# My fairly lazy OKHSV/OKHSL "port" of:
# https://github.com/holbrookdev/ok-color-picker
# MIT License

# Copyright (c) 2022 Brian Holbrook

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__ok_eps = 1e-7

def col_oklab_to_linear_srgb(lab:Union[Float3, Color]) -> Float3:
    """
    Convert OKLAB color to linear sRGB.

    :param lab: OKLAB color
    :type lab: Float3 | Color
    """
    L, a, b = lab.x, lab.y, lab.z
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.291485548 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    res = Float3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
    )
    return res

def col_linear_srgb_to_oklab(srgb:Union[Float3, Color]) -> Float3:
    """
    Convert linear sRGB color to OKLAB.

    :param srgb: linear sRGB color
    :type srgb: Float3 | Color
    """
    r, g, b = srgb.r, srgb.g, srgb.b
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    res = Float3(
        0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_
    )
    return res

def __ok_compute_max_saturation(ab:Float2) -> float:
    # Max saturation will be when one of r, g or b goes below zero.

    a, b = ab.x, ab.y

    # Select different coefficients depending on which component goes below zero first
    k0, k1, k2, k3, k4, wl, wm, ws = 0., 0., 0., 0., 0., 0., 0., 0.

    if -1.88170328 * a - 0.80936493 * b > 1:
        # Red component
        k0 = +1.19086277
        k1 = +1.76576728
        k2 = +0.59662641
        k3 = +0.75515197
        k4 = +0.56771245
        wl = +4.0767416621
        wm = -3.3077115913
        ws = +0.2309699292
    elif (1.81444104 * a - 1.19445276 * b > 1):
        # Green component
        k0 = +0.73956515
        k1 = -0.45954404
        k2 = +0.08285427
        k3 = +0.1254107
        k4 = +0.14503204
        wl = -1.2684380046
        wm = +2.6097574011
        ws = -0.3413193965
    else:
        # Blue component
        k0 = +1.35733652
        k1 = -0.00915799
        k2 = -1.1513021
        k3 = -0.50559606
        k4 = +0.00692167
        wl = -0.0041960863
        wm = -0.7034186147
        ws = +1.707614701

    # Approximate max saturation using a polynomial:
    S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

    # Do one step Halley's method to get closer
    # this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
    # this should be sufficient for most applications, otherwise do two/three steps

    k_l = +0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.291485548 * b

    l_ = 1 + S * k_l
    m_ = 1 + S * k_m
    s_ = 1 + S * k_s

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    l_dS = 3 * k_l * l_ * l_
    m_dS = 3 * k_m * m_ * m_
    s_dS = 3 * k_s * s_ * s_

    l_dS2 = 6 * k_l * k_l * l_
    m_dS2 = 6 * k_m * k_m * m_
    s_dS2 = 6 * k_s * k_s * s_

    f = wl * l + wm * m + ws * s
    f1 = wl * l_dS + wm * m_dS + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

    S = S - (f * f1) / np.max([f1 * f1 - 0.5 * f * f2, __ok_eps])

    return S

def __ok_find_cusp(ab:Float2) -> Float2:
    # First, find the maximum saturation (saturation S = C/L)
    S_cusp = __ok_compute_max_saturation(ab)

    # Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
    rgb_at_max = col_oklab_to_linear_srgb(Float3(1., S_cusp * ab.x, S_cusp * ab.y))

    L_cusp = np.cbrt(1. / np.max([np.max([rgb_at_max[0], rgb_at_max[1]]), rgb_at_max[2]]))
    C_cusp = L_cusp * S_cusp

    return Float2(L_cusp, C_cusp)



# Finds intersection of the line defined by
# L = L0 * (1 - t) + t * L1
# C = t * C1
# a and b must be normalized so a^2 + b^2 == 1
def __ok_find_gamut_intersection(
    a:float,
    b:float,
    L1:float,
    C1:float,
    L0:float,
    cusp:Float2=None) -> float:
    if cusp is None:
        # Find the cusp of the gamut triangle
        cusp = __ok_find_cusp(Float2(a, b))

    # Find the intersection for upper and lower half seprately
    t = 0.
    if ((L1 - L0) * cusp[1] - (cusp[0] - L0) * C1 <= 0):
        # Lower half
        t = (cusp[1] * L0) / (C1 * cusp[0] + cusp[1] * (L0 - L1))
    else:
        # Upper half
        # First intersect with triangle
        t = (cusp[1] * (L0 - 1)) / (C1 * (cusp[0] - 1) + cusp[1] * (L0 - L1))

        # Then one step Halley's method
        dL = L1 - L0
        dC = C1

        k_l = +0.3963377774 * a + 0.2158037573 * b
        k_m = -0.1055613458 * a - 0.0638541728 * b
        k_s = -0.0894841775 * a - 1.291485548 * b

        l_dt = dL + dC * k_l
        m_dt = dL + dC * k_m
        s_dt = dL + dC * k_s

        # If higher accuracy is required, 2 or 3 iterations of the following block can be used:

        L = L0 * (1 - t) + t * L1
        C = t * C1

        l_ = L + C * k_l
        m_ = L + C * k_m
        s_ = L + C * k_s

        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_

        ldt = 3 * l_dt * l_ * l_
        mdt = 3 * m_dt * m_ * m_
        sdt = 3 * s_dt * s_ * s_

        ldt2 = 6 * l_dt * l_dt * l_
        mdt2 = 6 * m_dt * m_dt * m_
        sdt2 = 6 * s_dt * s_dt * s_

        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
        r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
        r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2

        u_r = r1 / (r1 * r1 - 0.5 * r * r2)
        t_r = -r * u_r

        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1
        g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
        g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2

        u_g = g1 / (g1 * g1 - 0.5 * g * g2)
        t_g = -g * u_g

        b = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s - 1
        b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.707614701 * sdt
        b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.707614701 * sdt2

        u_b = b1 / (b1 * b1 - 0.5 * b * b2)
        t_b = -b * u_b

        t_r = t_r if u_r >= 0 else 10e5
        t_g = t_g if u_g >= 0 else 10e5
        t_b = t_b if u_b >= 0 else 10e5

        t += np.min([t_r, np.min([t_g, t_b])])
    return t

def __ok_get_st_max(ab:Float2, cusp=None) -> Float2:
    if cusp is None:
        # Find the cusp of the gamut triangle
        cusp = __ok_find_cusp(ab)
    L = cusp[0]
    C = cusp[1]
    return Float2(C / np.max([L, __ok_eps]), C / np.max([1 - L, __ok_eps]))

def __ok_toe(x:float) -> float:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1 + k_1) / (1 + k_2)
    return 0.5 * (k_3 * x -k_1 + np.sqrt((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x))

def __ok_toe_inv(x:float) -> float:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1 + k_1) / (1 + k_2)
    return (x * x + k_1 * x) / (k_3 * (x + k_2))

def __ok_get_cs(lab:Float3) -> Float3:
    L, a_, b_ = lab.x, lab.y, lab.z
    cusp = __ok_find_cusp(Float2(a_, b_))

    C_max = __ok_find_gamut_intersection(a_, b_, L, 1., L, cusp)
    ST_max = __ok_get_st_max(Float2(a_, b_), cusp)

    S_mid = 0.11516993 + 1 / (+7.4477897 + 4.1590124 * b_ + a_ * \
        (-2.19557347 +1.75198401 * b_ + a_ * \
            (-2.13704948 - 10.02301043 * b_ + a_ * \
                (-4.24894561 + 5.38770819 * b_ + 4.69891013 * a_))))

    T_mid = 0.11239642 + 1. / (+1.6132032 - 0.68124379 * b_ + a_ * \
        (+0.40370612 + 0.90148123 * b_ + a_ * \
            (-0.27087943 + 0.6122399 * b_ + a_ * \
                (+0.00299215 - 0.45399568 * b_ - 0.14661872 * a_))))

    k = C_max / np.max([np.min([L * ST_max[0], (1 - L) * ST_max[1]]), __ok_eps])

    C_mid = 0.
    C_a = L * S_mid
    C_b = (1 - L) * T_mid

    C_mid = 0.9 * k * np.sqrt(
        np.sqrt(1 / (1 /  np.max([  (C_a * C_a * C_a * C_a), __ok_eps ]) + \
            1 / np.max([(C_b * C_b * C_b * C_b), __ok_eps]))))

    C_0 = 0.
    C_a = L * 0.4
    C_b = (1 - L) * 0.8

    C_0 = np.sqrt(1 / (1 / np.max([(C_a * C_a) , __ok_eps]) + \
        1 / np.max([ (C_b * C_b), __ok_eps])))

    return Float3(C_0, C_mid, C_max)


def col_okhsv_to_srgb(hsv:Union[Float3, Color]) -> Float3:
    """
    Convert OKHSV color to linear sRGB.

    :param hsv: OKHSV color
    :type hsv: Float3 | Color
    """
    h, s, v = hsv.x, hsv.y, hsv.z
    a_ = np.cos(2 * np.pi * h)
    b_ = np.sin(2 * np.pi * h)

    ST_max = __ok_get_st_max(Float2(a_, b_))
    S_max = ST_max[0]
    S_0 = 0.5
    T = ST_max[1]
    k = 1 - S_0 / S_max

    L_v = 1 - (s * S_0) / (S_0 + T - T * k * s)
    C_v = (s * T * S_0) / (S_0 + T - T * k * s)

    L = v * L_v
    C = v * C_v


    L_vt = __ok_toe_inv(L_v)
    C_vt = (C_v * L_vt) / np.max([L_v, __ok_eps])

    L_new = __ok_toe_inv(L) # * L_v/L_vt
    C = (C * L_new) / np.max([L, __ok_eps])
    L = L_new

    rgb_scale = col_oklab_to_linear_srgb(Float3(L_vt, a_ * C_vt, b_ * C_vt))
    scale_L = np.cbrt(
        1 / np.max([rgb_scale[0], rgb_scale[1], rgb_scale[2], __ok_eps])
    )

    # remove to see effect without rescaling
    L = L * scale_L
    C = C * scale_L
    rgb = col_oklab_to_linear_srgb(Float3(L, C * a_, C * b_))

    return rgb.clip(0., 1.)

def col_srgb_to_okhsv(srgb:Union[Float3, Color]) -> Float3:
    """
    Convert linear sRGB color to OKHSV.

    :param srgb: linear sRGB color
    :type srgb: Float3 | Color
    """
    srgb = np.minimum(srgb, Float3(1. - __ok_eps))
    srgb = np.maximum(srgb, Float3(__ok_eps))
    lab = col_linear_srgb_to_oklab(srgb)

    C = np.sqrt(lab[1] * lab[1] + lab[2] * lab[2])
    a_ = lab[1] / np.max([C, __ok_eps])
    b_ = lab[2] / np.max([C, __ok_eps])
    L = lab[0]
    h = 0.5 + (0.5 * np.arctan2(-lab[2], -lab[1])) / np.pi

    ST_max = __ok_get_st_max(Float2(a_, b_))
    S_max = ST_max[0]
    S_0 = 0.5
    T = ST_max[1]
    k = 1 - S_0 / S_max

    t = T / np.max([C + L * T, __ok_eps])
    L_v = t * L
    C_v = t * C

    L_vt = __ok_toe_inv(L_v)
    C_vt = (C_v * L_vt) / np.max([L_v, __ok_eps])
    rgb_scale = col_oklab_to_linear_srgb(Float3(L_vt, a_ * C_vt, b_ * C_vt))
    scale_L = np.cbrt(
        1 / np.max([rgb_scale[0], rgb_scale[1], rgb_scale[2], __ok_eps])
    )

    L = L / scale_L
    C = C / scale_L

    C = (C * __ok_toe(L)) / np.max([L, __ok_eps])
    L = __ok_toe(L)

    v = L / np.max([L_v, __ok_eps])
    s = ((S_0 + T) * C_v) / np.max([T * S_0 + T * k * C_v, __ok_eps])

    return Float3(h, s, v).clip(0., 1.)

def col_okhsl_to_srgb(hsl:Union[Float3, Color]) -> Float3:
    """
    Convert OKHSL color to linear sRGB.

    :param hsl: OKHSL color
    :type hsl: Float3 | Color
    """
    h, s, l = hsl.x, hsl.y, hsl.z
    if l == 1:
        return Float3(255, 255, 255)
    elif l == 0:
        return Float3(0, 0, 0)

    a_ = np.cos(2 * np.pi * h)
    b_ = np.sin(2 * np.pi * h)
    L = __ok_toe_inv(l)

    Cs = __ok_get_cs(Float3(L, a_, b_))
    C_0 = Cs[0]
    C_mid = Cs[1]
    C_max = Cs[2]

    C, t, k_0, k_1, k_2 = 0., 0., 0., 0., 0.
    if s < 0.8:
        t = 1.25 * s
        k_0 = 0
        k_1 = 0.8 * C_0
        k_2 = 1 - k_1 / C_mid
    else:
        t = 5 * (s - 0.8)
        k_0 = C_mid
        k_1 = (0.2 * C_mid * C_mid * 1.25 * 1.25) / C_0
        k_2 = 1 - k_1 / (C_max - C_mid)

    C = k_0 + (t * k_1) / (1 - k_2 * t)

    # If we would only use one of the Cs:
    #C = s*C_0
    #C = s*1.25*C_mid
    #C = s*C_max

    rgb = col_oklab_to_linear_srgb(Float3(L, C * a_, C * b_))
    return rgb.clip(0., 1.)

def col_srgb_to_okhsl(srgb:Union[Float3, Color]) -> Float3:
    """
    Convert linear sRGB color to OKHSL.

    :param srgb: linear sRGB color
    :type srgb: Float3 | Color
    """
    srgb = np.minimum(srgb, Float3(1. - __ok_eps))
    srgb = np.maximum(srgb, Float3(__ok_eps))

    lab = col_linear_srgb_to_oklab(srgb)

    C = np.sqrt(lab[1] * lab[1] + lab[2] * lab[2])
    a_ = lab[1] / np.max([C, __ok_eps])
    b_ = lab[2] / np.max([C, __ok_eps])

    L = lab[0]
    h = 0.5 + (0.5 * np.arctan2(-lab[2], -lab[1])) / np.pi

    Cs = __ok_get_cs(Float3(L, a_, b_))
    C_0 = Cs[0]
    C_mid = Cs[1]
    C_max = Cs[2]

    s = 0.
    if C < C_mid:
        k_0 = 0
        k_1 = 0.8 * C_0
        k_2 = 1 - k_1 / np.max([C_mid, __ok_eps])

        t = (C - k_0) / np.max([(k_1 + k_2 * (C - k_0)), __ok_eps])
        s = t * 0.8
    else:
        k_0 = C_mid
        k_1 = (0.2 * C_mid * C_mid * 1.25 * 1.25) / np.max([C_0, __ok_eps])
        k_2 = 1 - k_1 / np.max([(C_max - C_mid), __ok_eps])

        t = (C - k_0) / np.max([(k_1 + k_2 * (C - k_0)), __ok_eps])
        s = 0.8 + 0.2 * t

    l = __ok_toe(L)
    return Float3(h, s, l).clip(0., 1.)
