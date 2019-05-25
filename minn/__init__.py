import weakref

import numpy as np


xp = np


# --------------------------------
# Computation Graph
# --------------------------------

class Graph(object):

    class Operation(object):
        __slots__ = ('f', 'args', 'rets')

        def __init__(self, f, args, rets):
            self.f, self.args, self.rets = f, tuple(args), tuple(rets)

    def __init__(self, inspect=True):
        self._ops = []
        self._inspect = inspect

    def clear(self):
        self._ops.clear()

    def apply(self, f, args):
        for v in args:
            self._check_var(v)
        x = [self._ops[v._oid].rets[v._vid].data for v in args]
        if self._inspect:
            f.check_forward(x)
        y_nodes = [VariableNode(data) for data in f.forward(x)]
        oid = len(self._ops)
        self._ops.append(Graph.Operation(f, args, y_nodes))
        return tuple(Variable(self, oid, i) for i in range(len(y_nodes)))

    def backprop(self, v):
        self._check_var(v)
        last_n = self._ops[v._oid].rets[v._vid]
        last_n.grad = xp.ones_like(last_n.data)
        for op in self._ops[v._oid::-1]:
            if all(node.grad is None for node in op.rets):
                continue
            y, gy = [], []
            for node in op.rets:
                if node.grad is None:
                    node.grad = xp.zeros_like(node.data)
                y.append(node.data)
                gy.append(node.grad)
            x, gx = [], []
            for v in op.args:
                node = self._ops[v._oid].rets[v._vid]
                if node.grad is None:
                    node.grad = xp.zeros_like(node.data)
                x.append(node.data)
                gx.append(node.grad)
            grads = op.f.backward(gy, x, y)
            assert len(grads) == len(gx)
            for i, grad in enumerate(grads):
                if grad is not None:
                    gx[i] += grad
            for node in op.rets:
                node.grad = None

    def get_data(self, v):
        self._check_var(v)
        return self._ops[v._oid].rets[v._vid].data

    def _check_var(self, v):
        if not isinstance(v, Variable):
            raise TypeError("object is not a `Variable`")
        if v._g() is not self:
            raise ValueError("graph mismatched")
        if v._oid >= len(self._ops) or v._vid >= len(self._ops[v._oid].rets):
            raise RuntimeError("invalid node")
        return


__default_graph = Graph()


class FunctionNode(object):

    def check_forward(self, x):
        return

    def forward(self, x):
        """return y"""
        raise NotImplementedError

    def backward(self, gy, x, y):
        """return gx"""
        raise NotImplementedError


class VariableNode(object):
    __slots__ = ('data', 'grad')

    def __init__(self, data):
        if not xp.isscalar(data):
            data.flags.writeable = False
        self.data = data
        self.grad = None


class Variable(object):
    def __init__(self, g, oid, vid):
        self._g = weakref.ref(g)
        self._oid = oid
        self._vid = vid

    def backward(self):
        return self._g().backprop(self)

    @property
    def data(self):
        return self._g().get_data(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return transpose(self)


def __get_graph(x=None):
    return x._g() if x is not None else __default_graph


def clear_graph():
    __default_graph.clear()


# --------------------------------
# Other Core classes
# --------------------------------

class Parameter(object):
    def __init__(self, shape):
        self.data = xp.empty(shape, dtype=xp.float32)
        self.grad = xp.zeros_like(self.data)
        self._initialized = False

    def initialize(self, initializer):
        if self._initialized:
            raise RuntimeError("this parameter has already been initialized")
        if isinstance(initializer, (int, float)):
            self.data[...] = initializer
        elif isinstance(initializer, xp.ndarray):
            if self.data.shape != initializer.shape:
                raise ValueError("shape mismatched")
            xp.copyto(self.data, initializer)
        else:
            initializer.initialize(self.data)
        self._initialized = True

    def reset_gradient(self):
        self.grad.fill(0.)

    @property
    def is_initialized(self):
        return self._initialized


class Model(object):

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Model)):
            self._params.add(name)
        super().__setattr__(name, value)

    def parameters(self):
        d = self.__dict__
        for name in sorted(self._params):
            prefix = '/' + name
            param = d[name]
            if isinstance(param, Model):
                for path, p in param.parameters():
                    yield prefix + path, param
            else:
                if not param.is_initialized:
                    raise RuntimeError(
                        "parameter '{}' is not initialized".format(prefix))
                yield prefix, param


class Initializer(object):

    def initialize(self, x):
        raise NotImplementedError


class Optimizer(object):

    def __init__(self):
        self._params = []

    def add(self, params):
        if isinstance(params, (list, tuple)):
            if any(not isinstance(p, Parameter) for p in params):
                raise TypeError("all elements of `params` must be `Parameter`")
            self._params.extend(params)
        elif isinstance(params, Model):
            self._params.extend(p for _, p in params.parameters())
        else:
            raise TypeError("invalid type")

    def update(self):
        raise NotImplementedError

    def reset_gradients(self):
        for p in self._params:
            p.reset_gradient()

    def _iter_params(self):
        for p in self._params:
            writeable = p.data.flags.writeable
            p.data.flags.writeable = True
            yield p
            p.data.flags.writeable = writeable


# --------------------------------
# FunctionNode implementations
# --------------------------------

class Input(FunctionNode):

    def __init__(self, data):
        self.data = data

    def forward(self, x):
        return self.data,

    def backward(self, gy, x, y):
        return []


def input(x):
    return __get_graph().apply(Input(x), [])[0]


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
        return []


def parameter(x):
    return __get_graph().apply(InputParameter(x), [])[0]


class Add(FunctionNode):

    def forward(self, x):
        x1, x2 = x
        return x1 + x2,

    def backward(self, gy, x, y):
        x1, x2 = x
        gy, = gy
        gx1 = gy * 1.
        gx2 = gy * 1.
        # TODO(chantera): support tensor (ndim >= 3)
        if gy.ndim == 2 and x1.ndim == 1:
            gx1 = xp.sum(gx1, axis=0)
        if gy.ndim == 2 and x2.ndim == 1:
            gx2 = xp.sum(gx2, axis=0)
        return gx1, gx2


def add(x1, x2):
    if not isinstance(x1, Variable):
        x1 = input(xp.array(x1, dtype=xp.float32))
    if not isinstance(x2, Variable):
        x2 = input(xp.array(x2, dtype=xp.float32))
    return __get_graph(x1).apply(Add(), (x1, x2))[0]


class Matmul(FunctionNode):

    def check_forward(self, x):
        x1, x2 = x
        assert x1.ndim == 2
        assert x2.ndim == 2

    def forward(self, x):
        x1, x2 = x
        return xp.dot(x1, x2),

    def backward(self, gy, x, y):
        x1, x2 = x
        gy, = gy
        gx1 = xp.dot(gy, x2.T)
        gx2 = xp.dot(x1.T, gy)
        return gx1, gx2


def matmul(x1, x2):
    return __get_graph(x1).apply(Matmul(), (x1, x2))[0]


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
    return __get_graph(x).apply(Transpose(), (x,))[0]


class Sigmoid(FunctionNode):

    def forward(self, x):
        x, = x
        half = x.dtype.type(0.5)
        y = xp.tanh(x * half) * half + half
        return y,

    def backward(self, gy, x, y):
        gy, = gy
        y, = y
        return gy * y * (1. - y),


def sigmoid(x):
    return __get_graph(x).apply(Sigmoid(), (x,))[0]


class Softmax(FunctionNode):

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        x, = x
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
    return __get_graph(x).apply(Softmax(axis), (x,))[0]


def _logsumexp(x, axis):
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
        gx = gy[0] - xp.exp(y[0]) * gy[0].sum(axis=self.axis, keepdims=True)
        return gx,


def log_softmax(x, axis=1):
    return __get_graph(x).apply(LogSoftmax(axis), (x,))[0]


class SoftmaxCrossEntropy:

    def check_forward(self, x):
        x, t = x
        assert x.ndim == 2
        assert t.ndim == 1

    def forward(self, x):
        x, t = x
        batch_size = t.shape[0]
        log_y = LogSoftmax().forward((x,))[0]
        y = -xp.sum(log_y[np.arange(batch_size), t]) / batch_size
        return y,

    def backward(self, gy, x, y):
        t = x[1]
        y, = y
        gx = Softmax().forward((x[0],))[0]
        gx[np.arange(t.shape[0]), t] -= 1
        return gx, None


def softmax_cross_entropy(x, t):
    if not isinstance(t, Variable):
        t = input(t)
    return __get_graph(x).apply(SoftmaxCrossEntropy(), (x, t))[0]


Variable.__add__ = add
Variable.__radd__ = add
Variable.__matmul__ = matmul


# --------------------------------
# Initializer implementations
# --------------------------------

class NormalInitializer(Initializer):

    def __init__(self, mean=0.0, sd=0.05):
        self.mean = mean
        self.sd = sd

    def initialize(self, x):
        x[...] = xp.random.normal(loc=self.mean, scale=self.sd, size=x.shape)


# --------------------------------
# Optimizer implementations
# --------------------------------

class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay_rate=0.001):
        super().__init__()
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate

    def update(self):
        for p in self._iter_params():
            p.data -= self.lr * p.grad + self.weight_decay_rate * p.data
