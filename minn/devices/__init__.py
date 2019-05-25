from minn.devices.cpu import CPU
from minn.devices.cpu import ndarray as _cpu_ndarray
from minn.devices.cuda import CUDA
from minn.devices.cuda import ndarray as _cuda_ndarray


def get_device_from_array(data):
    if isinstance(data, _cuda_ndarray):
        return CUDA.get_device(data)
    elif isinstance(data, _cpu_ndarray):
        return CPU.get_device(data)
    raise TypeError("invalid array")
