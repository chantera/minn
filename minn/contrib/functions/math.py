from minn.contrib import utils
from minn.contrib.devices import get_device_from_array
from minn.core import FunctionNode


class Add(FunctionNode):

    def forward(self, x):
        x1, x2 = x
        return utils.force_array(x1 + x2),

    def backward(self, gy, x, y):
        x1, x2 = x
        gy, = gy
        gx1 = gy * 1.
        gx2 = gy * 1.
        # TODO(chantera): support tensor (ndim >= 3)
        if gy.ndim == 2 and x1.ndim == 1:
            gx1 = gx1.sum(axis=0)
        if gy.ndim == 2 and x2.ndim == 1:
            gx2 = gx2.sum(axis=0)
        return gx1, gx2


class AddConstant(FunctionNode):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return utils.force_array(x[0] + x[0].dtype.type(self.value)),

    def backward(self, gy, x, y):
        return gy


def add(lhs, rhs):
    if utils.is_scalar(rhs):
        return lhs._g().apply(AddConstant(rhs), (lhs,))[0]
    return lhs._g().apply(Add(), (lhs, rhs))[0]


class Matmul(FunctionNode):

    def check_forward(self, x):
        x1, x2 = x
        assert x1.ndim == 2
        assert x2.ndim == 2

    def forward(self, x):
        x1, x2 = x
        xp = get_device_from_array(x1).xp
        return xp.dot(x1, x2),

    def backward(self, gy, x, y):
        x1, x2 = x
        xp = get_device_from_array(x1).xp
        gy, = gy
        gx1 = xp.dot(gy, x2.T)
        gx2 = xp.dot(x1.T, gy)
        return gx1, gx2


def matmul(x1, x2):
    return x1._g().apply(Matmul(), (x1, x2))[0]
