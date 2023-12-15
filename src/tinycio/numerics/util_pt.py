import typing
import torch
import torch.nn.functional as tf

class PTUtil:
    """PyTorch utility functions"""

    @staticmethod
    def lerp(a, b, w): return torch.lerp(a, b, w)

    @staticmethod
    def saturate(x): return x.clamp(0., 1.)

    @staticmethod
    def sign(x): y.sign()

    @staticmethod
    def normalize(v): return tf.normalize(v)

    @staticmethod
    def reflect(n, l):
        return l - (2. * torch.sum(l * n, dim=1).unsqueeze(-1) * n)