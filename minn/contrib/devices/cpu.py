import threading

import numpy
from numpy import ndarray  # NOQA

from minn.core import Device


class CPU(Device):
    __device = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls.__device is None:
                cls.__device = super().__new__(cls)
        return cls.__device

    def __init__(self):
        pass

    @property
    def xp(self):
        return numpy

    @classmethod
    def get_device(cls, array):
        return cls()
