import numpy as np
from math import sqrt
from typing import List, Tuple, Iterable
from ..tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params: List[Tensor] = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    momentum : float
        Momentum factor.
    weight_decay : float, default=0.
        Weight decay (L2 penalty).
    nesterov : bool, default=True.
        Whether to use Nesterov momentum.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov=True,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data
            self.v[i] *= self.momentum
            self.v[i] += self.lr * grad
            self.params[i].data -= self.v[i]
            if self.nesterov:
                self.params[i].data -= self.lr * grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            a_t = sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
            self.params[i].data -= (
                self.lr * a_t * self.m[i] / (self.v[i] ** 0.5 + self.eps)
            )
        self.t += 1
