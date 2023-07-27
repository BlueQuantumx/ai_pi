import numpy as np
from typing import Tuple, Union

from .. import tensor


def size_handle(input: Union[int, Tuple[int, int]]):
    if isinstance(input, int):
        return input, input
    assert type(input) in {list, tuple} and len(input) == 2
    return input


def linear(x: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine


class sigmoid(tensor.UnaryOperator):
    """Sigmoid Operation with Forward Propagation Avoiding Overflow"""

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        sigmoid = np.zeros(x.shape)
        sigmoid[x.data > 0] = 1 / (1 + np.exp(-x.data[x.data > 0]))
        sigmoid[x.data <= 0] = 1 - 1 / (1 + np.exp(x.data[x.data <= 0]))
        return sigmoid

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * (1 - self.data) * grad


class tanh(tensor.UnaryOperator):
    """Tanh Operation with Forward Propagation Avoiding Overflow"""

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        tanh = np.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + np.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + np.exp(2 * x.data[x.data <= 0]))
        return tanh

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        return (1 - self.data**2) * grad


def relu(x: tensor.Tensor):
    return tensor.maximum(0.0, x)


def leaky_relu(x: tensor.Tensor, alpha: float):
    return tensor.maximum(x, alpha * x)


def nll_loss(y_pred, y_true):
    """Negative Log Likelihood Loss"""
    nll = -tensor.log(y_pred) * y_true
    return tensor.mean(nll)


def binary_cross_entropy_loss(y_pred, y_true):
    """Binary Cross Entropy Loss"""
    nll = nll_loss(y_pred, y_true) + nll_loss(1 - y_pred, 1 - y_true)
    return tensor.mean(nll)
