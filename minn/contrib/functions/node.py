from minn.contrib import utils
from minn.contrib._internal import get_graph
from minn.core import FunctionNode
from minn.core import Parameter


class InputArray(FunctionNode):

    def __init__(self, data):
        if not utils.is_array(data):
            raise TypeError("input data must be an array")
        self.data = data

    def forward(self, x):
        return self.data,

    def backward(self, gy, x, y):
        return tuple()


def input(x, graph=None):
    return get_graph(graph).apply(InputArray(utils.force_array(x)), [])[0]


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
