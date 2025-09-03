from typing import Optional

from numNet import Tensor
from .. import functional as F
from ..types import _size_1_t, _size_2_t, _tuple_any_t
from .utils import _single, _pair
from .module import Module

class _MaxPoolNd(Module):
    def __init__(
        self,
        kernel_size: _tuple_any_t[int],
        stride: _tuple_any_t[int],
        padding: _tuple_any_t[int],
        dilation: _tuple_any_t[int],
        return_indices: bool = False
    ) -> None:
        super(_MaxPoolNd, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices


class MaxPool1d(_MaxPoolNd):
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False
    ):
        # Union[int, Tuple[int]] -> Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)

        super(MaxPool1d, self).__init__(
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            return_indices = return_indices
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )


class MaxPool2d(_MaxPoolNd):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False
    ):
        kernel_size_ = _pair(kernel_size)
        if stride:
            stride_ = _pair(stride)
        else:
            stride_ = kernel_size_
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super(MaxPool2d, self).__init__(
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            return_indices = return_indices
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )
