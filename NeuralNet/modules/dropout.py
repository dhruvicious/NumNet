from numNet import Tensor
from .module import Module
from .. import functional as F

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super(Dropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError(
                "Dropout probability has to be between 0 and 1, "
                "but got {}".format(p)
            )
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.dropout(input, self.p)
        return self.output
