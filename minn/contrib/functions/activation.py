from minn.contrib.devices import get_device_from_array
from minn.core import FunctionNode


class Sigmoid(FunctionNode):

    def forward(self, x):
        x, = x
        xp = get_device_from_array(x).xp
        half = x.dtype.type(0.5)
        y = xp.tanh(x * half) * half + half
        return y,

    def backward(self, gy, x, y):
        gy, = gy
        y, = y
        return gy * y * (1. - y),


def sigmoid(x):
    return x._g().apply(Sigmoid(), (x,))[0]


class Softmax(FunctionNode):

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        x, = x
        xp = get_device_from_array(x).xp
        y = x - x.max(axis=self.axis, keepdims=True)
        xp.exp(y, out=y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y,

    def backward(self, gy, x, y):
        gx = y[0] * gy[0]
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y[0] * sumdx
        return gx,


def softmax(x, axis=1):
    return x._g().apply(Softmax(axis), (x,))[0]


def _logsumexp(x, axis):
    xp = get_device_from_array(x).xp
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


class LogSoftmax(FunctionNode):

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        x, = x
        log_z = _logsumexp(x, self.axis)
        y = x - log_z
        return y,

    def backward(self, gy, x, y):
        xp = get_device_from_array(y[0]).xp
        gx = gy[0] - xp.exp(y[0]) * gy[0].sum(axis=self.axis, keepdims=True)
        return gx,


def log_softmax(x, axis=1):
    return x._g().apply(LogSoftmax(axis), (x,))[0]
