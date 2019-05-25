from minn.core import Optimizer


class SGD(Optimizer):

    def __init__(self, lr=0.01, weight_decay_rate=0.001):
        super().__init__()
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate

    def update(self):
        for p in self._iter_params():
            p.data -= self.lr * p.grad + self.weight_decay_rate * p.data
