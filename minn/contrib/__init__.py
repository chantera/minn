import minn
from minn.contrib import devices  # NOQA
from minn.contrib import functions  # NOQA
from minn.contrib import initializers  # NOQA
from minn.contrib import optimizers  # NOQA
from minn.contrib import utils  # NOQA
from minn.contrib.devices import get_device_from_array  # NOQA
from minn.contrib._internal import get_device  # NOQA
from minn.contrib._internal import set_device  # NOQA
from minn.contrib._internal import clear_graph  # NOQA
from minn.contrib._internal import get_graph  # NOQA
from minn.contrib._internal import set_graph  # NOQA


__all__ = [
    'clear_graph',
    'devices',
    'functions',
    'get_device',
    'get_device_from_array',
    'initializers',
    'optimizers',
    'get_graph',
    'set_device',
    'set_graph',
]


minn._internal.get_device = get_device
minn._internal.get_device_from_array = devices.get_device_from_array
functions._install_variable_methods()
set_device(devices.CPU())
set_graph(minn.core.Graph())
