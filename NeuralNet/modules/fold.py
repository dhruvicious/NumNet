from numNet import Tensor
from .. import functional as F
from ..types import _size_2_t
from .module import Module


class Unfold(Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1
    ) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        out, _, _ = F.unfold(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        return out
