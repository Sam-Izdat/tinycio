""" 
.. warning::

    The FloatX/IntX types here are Python wrappers around numpy arrays.
    They are glacially slow and should only be used for convenience 
    inside of Python scope and well outside of any high-traffic loops.

The numerics module offers shading-language-like syntax for float and int vector data types, 
as well as a few utility functions. These types are an extension of numpy arrays, and you 
can use them as such. Matrices are not natively supported.

.. highlight:: python
.. code-block:: python

    from tinycio.numerics import *
    import numpy as np
    import torch

    # Several possible inputs
    Float4(4,3,2,1)             # Float4([4., 3., 2., 1.])
    Float4(1.23)                # Float4([1.23, 1.23, 1.23, 1.23])
    Float4([4,3,2,1])           # Float4([4., 3., 2., 1.])
    Float4((4,3,2,1))           # Float4([4., 3., 2., 1.])
    Float4(np.array([3,2,1,0])) # Float4([3., 2., 1., 0.])
    Float4(torch.rand(4))       # Float4([0.22407186, 0.26193792, 0.89055574, 0.57386285])
    Float4(torch.rand(4,1,1))   # Float4([0.99545109, 0.46160549, 0.78145427, 0.02302521])

    # Swizzling
    foo = Float3(1,2,3)
    foo.y                       # 2.0 (float)
    foo.rg                      # Float2([1., 2.])
    foo.bgr                     # Float3([3., 2., 1.])
    foo.xxyx                    # Float4([1., 1., 2., 1.])
    
    # Utility functions
    bar = Float3.y_axis()       # Float3([0., 1., 0.])
    bar.list()                  # [0., 1., 0.]
    bar.tuple()                 # (0., 1., 0.)
    lerp(foo.bbb, bar, 0.5)     # Float3([1.5, 2. , 1.5])
    saturate(Float2(-2,5))      # Float2([0., 1.])
    sign(Float2(-2,5))          # Float2([-1.,  1.])
    Float4(1) == Float4.one()   # Float4([ True,  True,  True,  True])
"""
# User API
from .vector import Float2, Float3, Float4, Int2, Int3, Int4
from .linalg import lerp, saturate, sign, normalize, reflect, matmul_tl