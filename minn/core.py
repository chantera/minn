import weakref

from minn import _internal


class Graph(object):
    __slots__ = ('_ops', '_inspect', '__weakref__')

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
        outputs = f.forward(x)
        if self._inspect:
            self._check_forward_outputs(f, outputs)
        y_nodes = [VariableNode(data) for data in outputs]
        oid = len(self._ops)
        self._ops.append(Graph.Operation(f, args, y_nodes))
        return tuple(Variable(self, oid, i) for i in range(len(y_nodes)))

    def backprop(self, v):
        self._check_var(v)
        last_n = self._ops[v._oid].rets[v._vid]
        last_n.grad = last_n.xp.ones_like(last_n.data)
        for op in self._ops[v._oid::-1]:
            if all(node.grad is None for node in op.rets):
                continue
            x, gx, y, gy = self._gather_op_io(op)
            grads = op.f.backward(gy, x, y)
            if self._inspect:
                self._check_backward_outputs(op, grads)
            for i, grad in enumerate(grads):
                if grad is not None:
                    gx[i].append(grad)
            for node in op.rets:
                del node.grad
                node.grad = None

    def _gather_op_io(self, op):
        y, gy = [], []
        for node in op.rets:
            if node.grad is None:
                node.grad = node.xp.zeros_like(node.data)
            elif isinstance(node.grad, list):
                _reduce_grad(node)
            y.append(node.data)
            gy.append(node.grad)
        x, gx = [], []
        for v in op.args:
            node = self._ops[v._oid].rets[v._vid]
            if node.grad is None:
                node.grad = []
            x.append(node.data)
            gx.append(node.grad)
        return tuple(x), tuple(gx), tuple(y), tuple(gy)

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

    def _check_forward_outputs(self, f, outputs):
        if not isinstance(outputs, tuple):
            raise TypeError(
                "{}.forward must return tuple".format(f.__class__.__name__))

    def _check_backward_outputs(self, op, outputs):
        if not isinstance(outputs, tuple):
            raise TypeError(
                "{}.backward must return tuple"
                .format(op.f.__class__.__name__))
        elif len(outputs) != len(op.args):
            raise ValueError(
                "the size of outputs from {}.backward must be {}"
                .format(op.f.__class__.__name__, len(op.args)))


@profile
def _reduce_grad(node):
    if len(node.grad) > 0:
        grad = node.grad[0]
        if node.grad[0].shape != node.data.shape:
            grad = node.zeros_like(node.data) + grad
        for g in node.grad[1:]:
            if g.shape == grad.shape:
                grad += g
            else:
                grad = g + grad
        del node.grad
        node.grad = grad
    else:
        node.grad = None


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
    __slots__ = ('_device', 'data', 'grad')

    def __init__(self, data):
        self._device = weakref.ref(_internal.get_device_from_array(data))
        self.data = data
        self.grad = None

    @property
    def device(self):
        return self._device()

    @property
    def xp(self):
        return self.device.xp


class Variable(object):
    __slots__ = ('_g', '_oid', '_vid')

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


class Parameter(object):
    __slots__ = ('_device', 'data', 'grad', '_initialized')

    def __init__(self, shape, device=None):
        device = _internal.get_device(device)
        xp = device.xp
        self._device = weakref.ref(device)
        self.data = xp.empty(shape, dtype=xp.float32)
        self.grad = xp.zeros_like(self.data)
        self._initialized = False

    def initialize(self, initializer):
        if self._initialized:
            raise RuntimeError("this parameter has already been initialized")
        xp = self.xp
        if isinstance(initializer, (int, float)):
            self.data[...] = initializer
        elif isinstance(initializer, xp.ndarray):
            if self.data.shape != initializer.shape:
                raise ValueError("shape mismatched")
            xp.copyto(self.data, initializer)
        else:
            initializer.initialize(self)
        self._initialized = True

    def reset_gradient(self):
        self.grad.fill(0.)

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def device(self):
        return self._device()

    @property
    def shape(self):
        return self.data.shape

    @property
    def xp(self):
        return self.device.xp


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


class Device(object):

    @property
    def xp(self):
        raise NotImplementedError


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
