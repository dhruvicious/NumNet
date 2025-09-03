import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(
        self,
        params = None,
        lr: float = 0.01,
        eps: float = 1e-10,
        weight_decay: float = 0.
    ):
        super(Adagrad, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, p) in enumerate(zip(self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # accumulate squared gradients
                h += p.grad ** 2
                self.h[i] = h
                # update parameters
                p.data -= self.lr * p.grad / np.sqrt(h + self.eps)

        super(Adagrad, self).step()
