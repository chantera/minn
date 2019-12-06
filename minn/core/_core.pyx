import weakref

from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
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
            op.rets.push_back(Node(NULL))
        self._ops.push_back(op)
        cdef list outputs = [Variable.init(self, Address(oid, i), y[i])
                             for i in range(n_rets)]

        cdef Address addr
        cdef Node n
        if f._input_indexes_to_retain is not None:
            for i in f._input_indexes_to_retain:
                addr = op.args[i]
                n = self._ops[addr.oid].rets[addr.vid]
                if n.retained_data == NULL:
                    Py_INCREF(x[i])
                    n.retained_data = <PyObject*> x[i]
        if f._output_indexes_to_retain is not None:
            for i in f._output_indexes_to_retain:
                n = op.rets[i]
                if n.retained_data == NULL:
                    Py_INCREF(y[i])
                    n.retained_data = <PyObject*> y[i]

        return tuple(outputs)

    cdef Variable _as_variable(self, object data):
        if isinstance(data, Variable):
            return data
        cdef Operation op
        cdef size_t oid = self._ops.size()
        op.f = NULL
        op.rets.push_back(Node(NULL))
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
            Operation op
            Node n
        for op in self._ops:
            for n in op.rets:
                Py_XDECREF(n.retained_data)
            Py_XDECREF(op.f)
        self._ops.clear()

    def __dealloc__(self):
        self.clear()


cdef struct Node:
    PyObject *retained_data


cdef struct Address:
    size_t oid
    size_t vid


cdef struct Operation:
    PyObject *f
    vector.vector[Address] args
    vector.vector[Node] rets


cdef class Variable:
    cdef:
        Address _addr
        object _g, _data

    def __init__(self):
        raise RuntimeError('instantiation is prohibited')

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
