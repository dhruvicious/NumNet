import numpy as np
from .optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(
        self,
        params = None,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.
    ):
        super(RMSprop, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.alpha = alpha
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, p) in enumerate(zip(self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of the squared gradients
                h = self.alpha * h + (1 - self.alpha) * (p.grad ** 2)
                self.h[i] = h
                # update parameters
                p.data -= self.lr * p.grad / np.sqrt(h + self.eps)

        super(RMSprop, self).step()
