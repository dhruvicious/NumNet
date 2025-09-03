from numNet import Tensor
from .module import Module
from .. import functional as F

__all__ = [
    'Loss',
    'NllLoss',
    'CrossEntropyLoss',
    'MSELoss',
    'BCELoss'
]


class Loss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        self.reduction = reduction


class NllLoss(Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(NllLoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.nll_loss(input, target, reduction=self.reduction)
        return self.data


class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.cross_entropy(input, target, reduction=self.reduction)
        return self.data


class MSELoss(Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.mse_loss(input, target, reduction=self.reduction)
        return self.data


class BCELoss(Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.binary_cross_entropy(input, target, reduction=self.reduction)
        return self.data
