import numpy as np

from minn.core import FunctionNode


class Transpose(FunctionNode):

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        x, = x
        return x.transpose(self.axes),

    def backward(self, gy, x, y):
        gy, = gy
        inv_axes = self.axes
        if inv_axes is not None:
            axes_len = len(inv_axes)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in inv_axes]))
        return gy.transpose(inv_axes),


def transpose(x):
    return x._g().apply(Transpose(), (x,))[0]
