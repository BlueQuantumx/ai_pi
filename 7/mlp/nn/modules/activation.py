from .module import Module
from .. import functional as F
from ...tensor import Tensor


class Sigmoid(Module):
    """Activation Layer : Sigmoid"""

    def forward(self, x) -> Tensor:
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Tanh(Module):
    """Activation Layer : Tanh"""

    def forward(self, x) -> Tensor:
        return F.tanh(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class ReLU(Module):
    """Activation Layer : ReLU"""

    def forward(self, x) -> Tensor:
        return F.relu(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class LeakyReLU(Module):
    """
    Activation Layer : LeakyReLU

    alpha : float
        Slope of negative section.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x) -> Tensor:
        return F.leaky_relu(x, self.alpha)

    def __repr__(self) -> str:
        return "{}(alpha={})".format(self.__class__.__name__, self.alpha)
