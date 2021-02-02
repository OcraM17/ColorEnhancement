import torch
from torch.nn import functional as F
from utility import splines
"""
Ennet Network:
    parameters:
        -enhancement module: the module provided in input specifies the basis of the function that is combined 
            with the parameters (module.param_count) (as explained in the paper Learning Parametric Functions for Color Image Enhancement)
    The net takes the image in input and provide the parameters in output.
    The parameters are combined with the basis function and the learned color transformation is applied to the input image
    The enhancement module is the final layer of the net, it takes in input the parameters and combine it to provide the 
    color transformation. The color transformation then applied to the image.
    In this file there are several functions that could be combined with the parameters.
"""

class Ennet(torch.nn.Module):
    def __init__(self, enhancement_module):
        super().__init__()
        momentum = 0.01
        self.c1 = torch.nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=0)
        self.c2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.b2 = torch.nn.BatchNorm2d(16, momentum=momentum)
        self.c3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.b3 = torch.nn.BatchNorm2d(32, momentum=momentum)
        self.c4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.b4 = torch.nn.BatchNorm2d(64, momentum=momentum)
        self.downsample = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, enhancement_module.parameters_count)
        )
        self.emodule = enhancement_module

    def forward(self, image, applyto=None):
        x = image
        if (image.size(2), image.size(3)) != (256, 256):
            x = _bilinear(x, 256, 256)
        x = x - 0.5
        x = F.relu(self.c1(x))
        x = self.b2(F.leaky_relu(self.c2(x)))
        x = self.b3(F.leaky_relu(self.c3(x)))
        x = self.b4(F.leaky_relu(self.c4(x)))
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        if not self.training:
            result = torch.clamp(result, 0, 1)
        return result

class EnhancementModule(torch.nn.Module):
    def __init__(self, parameters_count):
        super().__init__()
        self.parameters_count = parameters_count

    def forward(self, image, parameters):
        return image


class FunctionBasis(EnhancementModule):
    def __init__(self, basis_dimension):
        super().__init__(basis_dimension * 3)
        self.bdim = basis_dimension

    def expand(self, x):
        """Bx3xHxW -> Bx3xDxHxW  where D is the dimension of the basis."""
        raise NotImplemented

    def forward(self, image, parameters):
        x = self.expand(image)
        w = parameters.view(parameters.size(0), 3, -1)
        return torch.einsum("bcfij,bcf->bcij", (x, w))


class PolynomialBasis(FunctionBasis):
    def __init__(self, dim):
        super().__init__(dim)
        exponents = torch.arange(dim).view(1, 1, -1, 1, 1).float()
        self.register_buffer("exponents", exponents)

    def expand(self, x):
        x = x.unsqueeze(2)
        return torch.pow(x, self.exponents)


class PiecewiseBasis(FunctionBasis):
    def __init__(self, dim):
        super().__init__(dim)
        nodes = torch.arange(dim).view(1, 1, -1, 1, 1).float()
        self.register_buffer("nodes", nodes)

    def expand(self, x):
        x = x.unsqueeze(2)
        return F.relu(1 - torch.abs((self.bdim - 1) * x - self.nodes))


class DCTBasis(FunctionBasis):
    def __init__(self, dim):
        super().__init__(dim)
        freqs = 6.283185307179586 * torch.arange(dim).view(1, 1, -1, 1, 1).float()
        self.register_buffer("freqs", freqs)

    def expand(self, x):
        x = x.unsqueeze(2)
        return torch.cos(x * self.freqs)


class RBFBasis(FunctionBasis):
    def __init__(self, dim, sigma=1.0):
        super().__init__(dim)
        nodes = torch.linspace(0, 1, dim).view(1, 1, -1, 1, 1)
        self.register_buffer("nodes", nodes)
        self.sigma = sigma / dim

    def expand(self, x):
        x = x.unsqueeze(2)
        # Somehow the usual 1/2 factor was missing
        return torch.exp(-((x - self.nodes) / self.sigma) ** 2)


class SeparableBasis(EnhancementModule):
    def __init__(self, basis1d):
        super().__init__((basis1d.bdim ** 3) * 3)
        self.basis1d = basis1d

    def expand3d(self, x):
        """Bx3xHxW -> Bx(D3)xHxW  where D3 is the cube of the dimension of the 1D basis."""
        e = self.basis1d.expand(x)
        e1 = torch.einsum("bdij,beij,bfij->bdefij", (e[:, 0, ...], e[:, 1, ...], e[:, 2, ...]))
        return e1.reshape(x.size(0), -1, x.size(2), x.size(3))

    def forward(self, image, parameters):
        x = self.expand3d(image)
        w = parameters.view(parameters.size(0), 3, -1)
        return torch.einsum("bdij,bcd->bcij", (x, w))


class Splines(EnhancementModule):
    def __init__(self, nodes):
        super().__init__(nodes * 3)
        self.interpolator = splines.SplineInterpolator(nodes)
        
    def forward(self, image, parameters):
        k = image.size(0) * 3
        x = image.view(k, -1)
        y = parameters.view(k, -1)
        z = self.interpolator(y, x)
        return z.view_as(image)
    

def _bilinear(im, height, width):
    xg = torch.linspace(-1, 1, width, device=im.device)
    yg = torch.linspace(-1, 1, height, device=im.device)
    mesh = torch.meshgrid([yg, xg])
    grid = torch.stack(mesh[::-1], 2).unsqueeze(0)
    grid = grid.expand(im.size(0), height, width, 2)
    return F.grid_sample(im, grid)


BASIS = {
    "splines" : Splines,
    "poly": PolynomialBasis,
    "pwise": PiecewiseBasis,
    "dct": DCTBasis,
    "rbf": RBFBasis,
    "poly3d": lambda p: SeparableBasis(PolynomialBasis(p)),
    "pwise3d": lambda p: SeparableBasis(PiecewiseBasis(p)),
    "dct3d": lambda p: SeparableBasis(DCTBasis(p)),
    "rbf3d": lambda p: SeparableBasis(RBFBasis(p))
}


def create_net(basis_name, basis_param):
    return Ennet(BASIS[basis_name](basis_param))


if __name__ == "__main__":
    net = create_net("splines", 3)
    print(net)
    x = torch.rand(7, 3, 256, 256)
    y = net(x)
    print(y.size())
    
