__default_device = None


def set_device(device):
    global __default_device
    __default_device = device


def get_device(device=None):
    return __default_device if device is None else device


__default_graph = None


def set_graph(graph):
    global __default_graph
    __default_graph = graph


def get_graph(graph=None):
    return __default_graph if graph is None else graph


def clear_graph():
    __default_graph.clear()
