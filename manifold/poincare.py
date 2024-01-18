import geoopt
import torch


class Poincare(geoopt.PoincareBall):
    def __init__(self, c=1.0, learnable=False):
        super(Poincare, self).__init__(c=c, learnable=learnable)

    def from_lorentz(self, x, dim=-1):
        x = x.to(self.c.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(self.c))

    def to_lorentz(self, x, dim=-1, eps=1e-6):
        x = x.to(self.c.device)
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.c)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res