import numpy as np

from minn.contrib import devices


def is_array(x):
    return isinstance(x, (devices.cpu.ndarray, devices.cuda.ndarray))


def is_scalar(x):
    return np.isscalar(x)


def force_array(x, dtype=None):
    if is_scalar(x):
        if dtype is None:
            return np.array(x)
        else:
            return np.array(x, dtype)
    else:
        if dtype is None:
            return x
        else:
            return x.astype(dtype, copy=False)
