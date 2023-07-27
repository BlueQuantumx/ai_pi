from .module import Module
from .. import functional as F
from ...tensor import Tensor


class Loss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class NLLLoss(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.nll_loss(y_pred, y_true)


class BinaryCrossEntropyLoss(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.binary_cross_entropy_loss(y_pred, y_true)
