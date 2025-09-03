import math
import numpy as np
from typing import Union, Tuple

import numNet
from ..tensor import Tensor
from ..utils import *
from .types import _tuple_1_t, _tuple_2_t, _tuple_any_t, _size_2_t
from .utils import im2col
from .modules.utils import _pair

def relu(input: Tensor) -> Tensor:
    out = Tensor(
        data = np.maximum(0., input.data),
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_relu():
        if input.requires_grad:
            input.grad += out.grad * ((input.data > 0) * np.ones_like(input.data))

    if out.requires_grad:
        out.grad_fn = grad_relu

    return out

def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    out = Tensor(
        data = np.maximum(negative_slope * input.data, input.data),
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_leaky_relu():
        if input.requires_grad:
            grad = np.ones_like(input.data)
            grad[input.data < 0] = negative_slope
            input.grad += out.grad * grad

    if out.requires_grad:
        out.grad_fn = grad_leaky_relu

    return out

def sigmoid(input: Tensor) -> Tensor:
    ret = 1 / (1 + np.exp(-input.data))

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_sigmoid():
        if input.requires_grad:
            input.grad += out.grad * out.data * (1 - out.data)

    if out.requires_grad:
        out.grad_fn = grad_sigmoid

    return out

def tanh(input: Tensor) -> Tensor:
    ret = np.tanh(input.data)

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_tanh():
        if input.requires_grad:
            input.grad += out.grad * (1 - np.square(out.data))

    if out.requires_grad:
        out.grad_fn = grad_tanh

    return out

def gelu(input: Tensor) -> Tensor:
    out = 0.5 * input * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input ** 3.0))))
    return out

def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    dim = input.ndim

    if dim != 2:
        raise ValueError("Expected 2 dimensions (got {})".format(dim))

    if input.shape[0] != target.shape[0]:
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(input.shape[0], target.shape[0])
        )

    batch_size = input.shape[0]
    n_classes = input.shape[1]
    delta = 1e-7

    ret = - np.log(input.data[np.arange(batch_size), target.data.astype(np.int)] + delta)
    if reduction in ['sum', 'mean']:
        ret = np.sum(ret)
    if reduction == 'mean':
        ret = ret / batch_size

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_nll():
        if input.requires_grad:
            p = np.clip(input.data, 1e-15, 1 - 1e-15)
            y = to_categorical(target.data, n_classes=n_classes)
            if reduction == 'mean':
                input.grad += (p - y) / batch_size
            elif reduction == 'sum':
                input.grad += (p - y)

    if out.requires_grad and reduction != 'none':
        out.grad_fn = grad_nll

    return out

def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    after_softmax = input.softmax(dim=-1)
    out = nll_loss(after_softmax, target, reduction)

    return out

def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.numel

    out = (input - target) ** 2
    if reduction in ['sum', 'mean']:
        out = out.sum()
    if reduction == 'mean':
        out = out / n

    return out

def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.numel

    out = - (target * input.log() + (-target + 1.) * (-input + 1.).log())
    if reduction in ['sum', 'mean']:
        out = out.sum()
    if reduction == 'mean':
        out = out / n

    return out

def pad(input: Tensor, pad: _tuple_any_t[int], value: int = 0) -> Tensor:
    n_pad_dims = int(len(pad) / 2)
    ndims = input.ndim

    no_pad_width = [(0, 0) for i in range(0, ndims - n_pad_dims)]
    pad_width = no_pad_width + [(pad[i * 2], pad[i * 2 + 1]) for i in range(0, n_pad_dims)]

    ret = np.pad(
        input.data,
        pad_width = pad_width,
        mode = 'constant',
        constant_values = value,
    )

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def unpad(x: Tensor):
        slices = [slice(p[0], None if p[1] == 0 else -p[1]) for p in pad_width]
        return x[tuple(slices)]

    def grad_pad():
        if input.requires_grad:
            input.grad += unpad(out.grad)

    if out.requires_grad:
        out.grad_fn = grad_pad

    return out

def linear(input: Tensor, weight: Tensor, bias: Tensor = None):
    out = input @ weight

    if bias is not None:
        out += bias

    return out

def unfold(
    input: Tensor,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 1
):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    batch_size, in_channels, h_in, w_in = input.shape
    kernel_h, kernel_w = kernel_size

    h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1)

    padded_data = pad(input, (0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]))

    unfolded = im2col(padded_data, kernel_size, (h_out, w_out), stride, dilation)  # (batch_size, kernel_h * kernel_w * in_channels, L = h_out * w_out)

    return unfolded, h_out, w_out

def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: _tuple_2_t[int] = (1, 1),
    padding: _tuple_2_t[int] = (0, 0),
    dilation: _tuple_2_t[int] = (1, 1)
):
    batch_size, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape

    input_col, h_out, w_out = unfold(input, (kernel_h, kernel_w), stride, padding, dilation)
    input_col = input_col.permute(1, 2, 0).view(kernel_h * kernel_w * in_channels, -1)  # (kernel_h * kernel_w * in_channels, batch_size * h_out * w_out)

    weight_col = weight.view(out_channels, -1)

    out = (weight_col @ input_col).view(out_channels, h_out, w_out, batch_size).permute(3, 0, 1, 2)

    if bias is not None:
        out += bias

    return out


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: _tuple_1_t[int] = (1, ),
    padding: _tuple_1_t[int] = (0, ),
    dilation: _tuple_1_t[int] = (1, )
):
    input_2d = input.unsqueeze(dim=2)
    weight_2d = weight.unsqueeze(dim=2)
    bias_2d = bias.unsqueeze(dim=2)

    stride_2d = (1, stride[0])
    pad_2d = (0, padding[0])
    dilation_2d = (1, dilation[0])

    out_2d = conv2d(input_2d, weight_2d, bias_2d, stride_2d, pad_2d, dilation_2d)  # (batch_size, out_channels, 1, L_out)

    out = out_2d.squeeze(dim=2)
    return out

def max_pool2d(
    input: Tensor,
    kernel_size: _tuple_2_t[int],
    stride: _tuple_2_t[int],
    padding: _tuple_2_t[int] = (0, 0),
    dilation: _tuple_2_t[int] = (1, 1),
    return_indices: bool = False
):
    batch_size, in_channels, h_in, w_in = input.shape
    kernel_h, kernel_w = kernel_size

    input_col, h_out, w_out = unfold(input, kernel_size, stride, padding, dilation)
    input_col = input_col.permute(1, 2, 0).view(in_channels, kernel_h * kernel_w, -1)

    out_max = input_col.max(dim=1).view(in_channels, h_out, w_out, batch_size).permute(3, 0, 1, 2)
    return out_max

def max_pool1d(
    input: Tensor,
    kernel_size: _tuple_1_t[int],
    stride: _tuple_1_t[int] = (1, ),
    padding: _tuple_1_t[int] = (0, ),
    dilation: _tuple_1_t[int] = (1, ),
    return_indices: bool = False
):
    input_2d = input.unsqueeze(dim=2)

    kernel_size_2d = (1, kernel_size)
    stride_2d = (1, stride[0])
    pad_2d = (0, padding[0])
    dilation_2d = (1, dilation[0])

    out_2d = max_pool2d(input_2d, kernel_size_2d, stride_2d, pad_2d, dilation_2d, return_indices)  # (batch_size, out_channels, 1, L_out)

    out = out_2d.squeeze(dim=2)
    return out

def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    ret = input.data
    scaler = 1.0 / (1.0 - p)
    mask = np.random.binomial(1, 1 - p, size=input.shape)

    if training:
        ret = scaler * mask * ret

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_dropout():
        if input.requires_grad:
            input.grad += scaler * mask * out.grad

    if out.requires_grad:
        out.grad_fn = grad_dropout

    return out

def flatten(input: Tensor) -> Tensor:
    return input.view(input.size(0), -1)
