from numNet import Tensor
from .module import Module
from .. import functional as F

class Flatten(Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.flatten(input)
        return self.output
