import geoopt
import torch


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(self.k))