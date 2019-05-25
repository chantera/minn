from minn.core import Initializer


class NormalInitializer(Initializer):

    def __init__(self, mean=0.0, sd=0.05):
        self.mean = mean
        self.sd = sd

    def initialize(self, x):
        x.data[...] = x.xp.random.normal(
            loc=self.mean, scale=self.sd, size=x.shape)
