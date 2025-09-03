from numNet import Tensor
from .. import functional as F
from .module import Module

__all__ = [
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
    'GELU'
]


class ReLU(Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.relu(input)
        return self.data


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        super(ReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.leaky_relu(input, self.negative_slope)
        return self.data


class Sigmoid(Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.sigmoid(input)
        return self.data


class Tanh(Module):
    def __init__(self) -> None:
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.tanh(input)
        return self.data


class GELU(Module):
    def __init__(self) -> None:
        super(GELU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.gelu(input)
        return self.data
