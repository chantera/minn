import numpy as np

from minn.contrib._internal import get_graph
from minn.contrib.devices import get_device_from_array
from minn.core import FunctionNode, Parameter, Variable


def _force_array(x, dtype=None):
    if np.isscalar(x):
        if dtype is None:
            return np.array(x)
        else:
            return np.array(x, dtype)
    else:
        if dtype is None:
            return x
        else:
            return x.astype(dtype, copy=False)


class Input(FunctionNode):

    def __init__(self, data):
        self.data = data

    def forward(self, x):
        return self.data,

    def backward(self, gy, x, y):
        return tuple()


def input(x, graph=None):
    return get_graph(graph).apply(Input(_force_array(x)), [])[0]


class InputParameter(FunctionNode):

    def __init__(self, param):
        if not isinstance(param, Parameter):
            raise TypeError("`param` is not a `Parameter`")
        elif not param.is_initialized:
            raise ValueError("`param` must be initialized")
        self.param = param

    def forward(self, x):
        return self.param.data,

    def backward(self, gy, x, y):
        self.param.grad += gy[0]
        return tuple()


def parameter(x, graph=None):
    return get_graph(graph).apply(InputParameter(x), [])[0]


class Add(FunctionNode):

    def forward(self, x):
        x1, x2 = x
        return _force_array(x1 + x2),

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
        return _force_array(x[0] + x[0].dtype.type(self.value)),

    def backward(self, gy, x, y):
        return gy


def add(lhs, rhs):
    if np.isscalar(rhs):
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


def _install_variable_methods():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__matmul__ = matmul
    Variable.T = property(transpose)


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


class SoftmaxCrossEntropy:

    def check_forward(self, x):
        x, t = x
        assert x.ndim == 2
        assert t.ndim == 1

    def forward(self, x):
        x, t = x
        batch_size = t.shape[0]
        log_y = LogSoftmax().forward((x,))[0]
        log_p = log_y[np.arange(batch_size), t]
        y = -log_p.sum(keepdims=True) / batch_size
        return y.reshape(()),

    def backward(self, gy, x, y):
        t = x[1]
        y, = y
        gx = Softmax().forward((x[0],))[0]
        gx[np.arange(t.shape[0]), t] -= 1
        return gx, None


def softmax_cross_entropy(x, t):
    if not isinstance(t, Variable):
        t = input(t)
    return x._g().apply(SoftmaxCrossEntropy(), (x, t))[0]
