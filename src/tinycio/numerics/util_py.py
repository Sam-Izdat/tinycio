import typing
import numpy as np

class PyUtil:
    """Vanilla python/numpy utility functions"""

    @staticmethod
    def lerp(a, b, w):
        if isinstance(a, list) and isinstance(b, list):
            return [lerp(a[idx], b[idx], w) for idx, _ in enumerate(a)]
        return a * (1. - w) + b * w
        
    @staticmethod
    def saturate(x):
        if isinstance(x, np.ndarray): 
            return np.clip(x, 0, 1)
        else:
            if x < 0.: return 0.
            elif x > 1.: return 1.
            return x

    @staticmethod
    def sign(x):
        if isinstance(x, np.ndarray): 
            return np.sign(x)
        else:
            if x < 0.: return -1.
            elif x > 0.: return 1.
            return 0.

    @staticmethod
    def normalize(v):
        return v/(np.linalg.norm(v, axis=0, keepdims=True)+1e-10)

    @staticmethod
    def reflect(n, l):
        return 2. * np.dot(n, l) * n - l