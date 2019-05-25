from minn.contrib.functions.activation import log_softmax  # NOQA
from minn.contrib.functions.activation import sigmoid  # NOQA
from minn.contrib.functions.activation import softmax  # NOQA
from minn.contrib.functions.array import transpose  # NOQA
from minn.contrib.functions.loss import softmax_cross_entropy  # NOQA
from minn.contrib.functions.math import add  # NOQA
from minn.contrib.functions.math import matmul  # NOQA
from minn.contrib.functions.node import input  # NOQA
from minn.contrib.functions.node import parameter  # NOQA
from minn.core import Variable


def _install_variable_methods():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__matmul__ = matmul
    Variable.T = property(transpose)
