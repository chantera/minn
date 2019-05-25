import numpy as np

from minn.contrib.functions.activation import LogSoftmax
from minn.contrib.functions.activation import Softmax
from minn.core import FunctionNode
from minn.core import Variable


class SoftmaxCrossEntropy(FunctionNode):

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
