import numpy as np
from typing import Tuple
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(
        self,
        params = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.
    ):
        super(Adam, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        super(Adam, self).step()

        for i, (v, h, p) in enumerate(zip(self.v, self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of gradients
                v = self.beta1 * v + (1 - self.beta1) * p.grad
                self.v[i] = v
                # moving average of squared gradients
                h = self.beta2 * h + (1 - self.beta2) * (p.grad ** 2)
                self.h[i] = h
                # bias correction
                v_correction = 1 - (self.beta1 ** self.iterations)
                h_correction = 1 - (self.beta2 ** self.iterations)
                # update parameters
                p.data -= (self.lr / v_correction * v) / (np.sqrt(h) / np.sqrt(h_correction) + self.eps)
