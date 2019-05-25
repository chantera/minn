from minn import devices  # NOQA
from minn import functions  # NOQA
from minn import initializers  # NOQA
from minn import optimizers  # NOQA
from minn.core import Device  # NOQA
from minn.core import FunctionNode  # NOQA
from minn.core import Graph  # NOQA
from minn.core import Initializer  # NOQA
from minn.core import Model  # NOQA
from minn.core import Optimizer  # NOQA
from minn.core import Parameter  # NOQA
from minn.core import Variable  # NOQA
from minn.devices import get_device_from_array  # NOQA
from minn._internal import get_device  # NOQA
from minn._internal import set_device  # NOQA
from minn._internal import clear_graph  # NOQA
from minn._internal import get_graph  # NOQA
from minn._internal import set_graph  # NOQA


functions._install_variable_arithmetics()
set_device(devices.CPU())
set_graph(Graph())
