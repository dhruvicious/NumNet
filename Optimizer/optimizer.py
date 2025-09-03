class Optimizer:
    def __init__(self, params = None, lr: float = 0.01, weight_decay: float = 0.):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not weight_decay >= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.iterations += 1
