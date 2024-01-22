import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z