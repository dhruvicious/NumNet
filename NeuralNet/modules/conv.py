import math
from typing import Union, Tuple

from numNet import Tensor
from .. import Parameter, init
from .. import functional as F
from ..types import _size_1_t, _size_2_t, _tuple_any_t
from .utils import _single, _pair
from .module import Module


class _ConvNd(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _tuple_any_t[int],
        stride: _tuple_any_t[int],
        padding: _tuple_any_t[int],
        dilation: _tuple_any_t[int],
        bias: bool
    ) -> None:
        super(_ConvNd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Parameter(Tensor.zeros(out_channels, in_channels, *kernel_size))

        if bias:
            self.bias = Parameter(Tensor.zeros(1, out_channels, *[1 for k in kernel_size]))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        bias: bool = True
    ):
        # Union[int, Tuple[int]] -> Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)

        super(Conv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            bias = bias
        )

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )
        return self.output


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias: bool = True
    ):
        # Union[int, Tuple[int, int]] -> Tuple[int]
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super(Conv2d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            bias = bias
        )

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )
        return self.output
