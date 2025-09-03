import math
import numpy as np
from typing import Optional, Union

from numNet import Tensor

def calculate_gain(nonlinearity: str, param: Optional[Union[int, float]] = None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def zeros_(tensor: Tensor) -> None:
    tensor.zero_()

def ones_(tensor: Tensor) -> None:
    tensor.one_()

def constant_(tensor: Tensor, val: float) -> None:
    tensor.fill_(val)

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> None:
    tensor.uniform_(low=a, high=b)

def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> None:
    tensor.normal_(mean=mean, std=std)


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = np.prod(tensor.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # calculate uniform bounds from standard deviation

    tensor.uniform_(low=-a, high=a)

def xavier_normal_(tensor: Tensor, gain: float = 1.) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    tensor.normal_(mean=0, std=std)


def _calculate_correct_fan(tensor: Tensor, mode: str):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    tensor.uniform_(low=-bound, high=bound)

def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    tensor.normal_(mean=0, std=std)


def lecun_uniform_(tensor: Tensor) -> None:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(3.0 / fan_in)

    tensor.uniform_(low=-bound, high=bound)

def lecun_normal_(tensor: Tensor) -> None:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(1.0 / fan_in)

    tensor.normal_(mean=0, std=std)
