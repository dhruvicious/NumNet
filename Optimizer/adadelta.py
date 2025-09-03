import numpy as np
from .optimizer import Optimizer

class Adadelta(Optimizer):
    def __init__(
        self,
        params = None,
        rho: float = 0.99,
        eps: float = 1e-6,
        lr: float = 1.0,
        weight_decay: float = 0.
    ):
        super(Adadelta, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.rho = rho
        self.h = [np.zeros_like(p.data) for p in self.params]
        self.delta = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, delta, p) in enumerate(zip(self.h, self.delta, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of the squared gradients
                h = self.rho * h + (1 - self.rho) * (p.grad ** 2)
                self.h[i] = h
                # compute g'_t and delta_t
                g_ = np.sqrt(delta + self.eps) / np.sqrt(h + self.eps) * p.grad
                delta = self.rho * delta + (1 - self.rho) * (g_ ** 2)
                self.delta[i] = delta
                # update parameters
                p.data -= self.lr * g_

        super(Adadelta, self).step()
