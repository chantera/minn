import threading

from minn.core import Device

available = False

try:
    import cupy
    from cupy import ndarray  # NOQA

    available = True
except Exception as e:
    _resolution_error = e

    class ndarray(object):
        pass


class CUDA(Device):
    __devices = {}
    _lock = threading.Lock()

    def __new__(cls, device_id):
        with cls._lock:
            device = cls.__devices.get(device_id, None)
            if device is None:
                if not available:
                    raise RuntimeError("CUDA device is not available")
                device = super().__new__(cls)
                device._dev = cupy.Device(device_id)
                cls.__devices[device_id] = device
            return device

    def __init__(self, device_id):
        pass

    @property
    def xp(self):
        self._dev.use()
        return cupy

    @classmethod
    def get_device(cls, array):
        if not isinstance(array, ndarray):
            raise TypeError("invalid cupy array")
        return cls(array.device.id)
