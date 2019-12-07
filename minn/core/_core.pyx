import weakref

from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libcpp.vector cimport vector


cdef class Graph:
    cdef:
        vector[Operation] _ops
        object __weakref__

    def __init__(self):
        pass

    cpdef tuple apply(self, Function f, tuple args):
        cdef inputs = list(args)
        cdef Py_ssize_t i
        for i, v in enumerate(inputs):
            if isinstance(v, Variable):
                self._check_var(v)
            else:
                inputs[i] = self._as_variable(v)
        return self._apply(f, inputs)

    cdef tuple _apply(self, Function f, list args):
        cdef Variable v
        cdef list x = [v._data for v in args]
        cdef tuple y = f.forward(x)

        cdef Operation op
        cdef size_t oid = self._ops.size()
        cdef Py_ssize_t n_args = len(args)
        cdef Py_ssize_t n_rets = len(y)
        cdef Py_ssize_t i
        Py_INCREF(f)
        op.f = <PyObject*> f
        op.args.reserve(n_args)
        for i in range(n_args):
            op.args.push_back((<Variable> args[i])._addr)
        op.rets.reserve(n_rets)
        for i in range(n_rets):
            _push_node_to_op(&op, <PyObject*> y[i])
        self._ops.push_back(op)
        cdef list outputs = [Variable.init(self, Address(oid, i), y[i])
                             for i in range(n_rets)]

        cdef Address *addr
        cdef Node *n
        if f._input_indexes_to_retain is not None:
            for i in f._input_indexes_to_retain:
                addr = &op.args[i]
                n = &self._ops[addr.oid].rets[addr.vid]
                if n.retained_data == NULL:
                    Py_INCREF(x[i])
                    n.retained_data = <PyObject*> x[i]
        if f._output_indexes_to_retain is not None:
            for i in f._output_indexes_to_retain:
                n = &op.rets[i]
                if n.retained_data == NULL:
                    Py_INCREF(y[i])
                    n.retained_data = <PyObject*> y[i]

        return tuple(outputs)

    cpdef list backprop(self, Variable v, list srcs):
        return self._gradients(v, srcs)

    cdef list _gradients(self, Variable v, list srcs):
        # NOTE: high order differentiation is not currently supported
        self._check_var(v)
        cdef Py_ssize_t n_srcs = len(srcs)
        if n_srcs == 0:
            return []

        cdef vector[DataId] src_ids = vector[DataId](n_srcs)
        cdef Py_ssize_t i
        for i in range(n_srcs):
            src_ids.push_back(id(srcs[i]))
        cdef set req_ids = set(src_ids)
        cdef dict req_grads = dict.fromkeys(req_ids, None)

        cdef vector[Operation] *ops = &self._ops
        cdef Py_ssize_t oid
        cdef Operation *op
        cdef Py_ssize_t n_args
        cdef Py_ssize_t n_rets
        cdef list x, gx, y, gy

        cdef dict grads = {}
        cdef list out_grads
        cdef object grad
        cdef Node *node = &ops.at(v._addr.vid).rets[v._addr.vid]
        grad = (<object> node.device)().ones(node.shape, <object> node.dtype)
        grads[(v._addr.oid, v._addr.vid)] = grad

        # full backprop: this may compute gradients for unneeded nodes
        for oid in range(v._addr.oid, -1, -1):
            op = &ops.at(oid)
            n_args = op.args.size()
            n_rets = op.rets.size()

            if all(grads.get((oid, i)) is None for i in range(n_rets)):
                continue
            x, gx, y, gy = Graph._gather_op_io(ops, oid, grads)
            if op.f != NULL:
                out_grads = (<object> op.f).backward(gy, x, y)
            else:
                assert n_args == 0
                out_grads = []
            if n_args != len(out_grads):
                raise ValueError(
                    "the size of outputs from {}.backward must be {}"
                    .format((<object> op.f).__class__.__name__, n_args))
            for i in range(n_args):
                grad = out_grads[i]
                if grad is not None:
                    gx[i].append(grad)
            for i in range(n_rets):
                data_id = op.rets[i].data_id
                if data_id in req_ids:
                    grad = grads[(oid, i)]
                    if req_grads[data_id] is None:
                        req_grads[data_id] = grad
                    else:
                        req_grads[data_id] += grad
                del grads[(oid, i)]

        # del grads
        return list(req_grads.values())

    @staticmethod
    cdef tuple _gather_op_io(vector[Operation] *ops, size_t oid, dict grads):
        cdef list x = [], gx = [], y = [], gy = []
        cdef Operation *op = &ops.at(oid)
        cdef Py_ssize_t n_args = op.args.size()
        cdef Py_ssize_t n_rets = op.rets.size()
        cdef Node *node
        cdef Address *addr
        for i in range(n_rets):
            node = &op.rets[i]
            grad = grads.get((oid, i))
            if grad is None:
                grad = (<object> node.device)().zeros(
                    node.shape, <object> node.dtype)
                grads[(oid, i)] = grad
            elif isinstance(grad, list):
                # TODO: implement reduce
                pass
            y.append((<object> node.retained_data)
                    if node.retained_data != NULL else None)
            gy.append(grad)
        for i in range(n_args):
            addr = &op.args[i]
            node = &ops.at(addr.oid).rets[addr.vid]
            grad = grads.get((addr.oid, addr.vid))
            if grad is None:
                grad = []
                grads[(addr.oid, addr.vid)] = grad
            x.append((<object> node.retained_data)
                    if node.retained_data != NULL else None)
            gx.append(grad)
        return x, gx, y, gy

    cdef Variable _as_variable(self, object data):
        if isinstance(data, Variable):
            return data
        cdef Operation op
        cdef size_t oid = self._ops.size()
        op.f = NULL
        _push_node_to_op(&op, <PyObject*> data)
        self._ops.push_back(op)
        return Variable.init(self, Address(oid, 0), data)

    cdef void _check_var(self, Variable v):
        if v._g() is not self:
            raise ValueError("graph mismatched")
        if v._addr.oid >= self._ops.size() \
                or v._addr.vid >= self._ops[v._addr.oid].rets.size():
            raise RuntimeError("invalid node")
        return

    def clear(self):
        cdef:
            vector[Node] *rets
            vector[Node].iterator n_it
            Node *node
            vector[Operation].iterator op_it = self._ops.begin()
        while op_it != self._ops.end():
            rets = &deref(op_it).rets
            n_it = rets.begin()
            while n_it != rets.end():
                node = &deref(n_it)
                Py_XDECREF(node.dtype)
                Py_XDECREF(node.device)
                Py_XDECREF(node.retained_data)
                preinc(n_it)
            Py_XDECREF(deref(op_it).f)
            preinc(op_it)
        self._ops.clear()

    def __dealloc__(self):
        self.clear()


"""
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
"""


ctypedef long DataId


cdef struct Node:
    vector.vector[int] shape
    PyObject *dtype
    PyObject *device
    DataId data_id
    PyObject *retained_data
    PyObject *grad


cdef struct Address:
    size_t oid
    size_t vid


cdef struct Operation:
    PyObject *f
    vector.vector[Address] args
    vector.vector[Node] rets


cdef void _push_node_to_op(Operation *op, PyObject *data):
    cdef object d = <object> data
    cdef object dtype = d.dtype
    cdef object device = weakref.ref(get_device(d))
    Py_INCREF(dtype)
    Py_INCREF(device)
    op.rets.push_back(Node(
        d.shape, <PyObject*> dtype, <PyObject*> device, id(d), NULL, NULL))


cdef class Variable:
    cdef:
        Address _addr
        object _g, _data

    def __init__(self):
        raise RuntimeError("instantiation is prohibited")

    @staticmethod
    cdef Variable init(Graph g, Address addr, object data):
        cdef Variable v = Variable.__new__(Variable)
        v._g = weakref.ref(g)
        v._addr = addr
        v._data = data
        return v

    @property
    def data(self):
        return self._data


cdef object _device_getter = None


def set_device_getter(f):
    _device_getter = f


cpdef object get_device(object data):
    if _device_getter is None:
        raise RuntimeError("`device_getter` has not been set")
    return _device_getter(data)


cdef class Function:
    _input_indexes_to_retain = None
    _output_indexes_to_retain = None

    def apply(self, tuple args, Graph g=None):
        if g is not None:
            return g.apply(self, args)
        for v in args:
            if isinstance(v, Variable):
                return (<Variable> v)._g().apply(self, args)
        y = self.forward(args)
        return tuple(y)

    def forward(self, tuple x):
        raise NotImplementedError

    def backward(self, tuple gy, tuple x, tuple y):
        raise NotImplementedError

    def retain_inputs(self, indexes):
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes):
        self._output_indexes_to_retain = indexes
