from typing import Any, List, Tuple, Union, Optional
import numpy as np

from .autograd import Graph, no_grad, is_grad_enable


class Tensor:
    """
    Wrapper of NumPy array, with support of autograd.

    Parameters
    ----------
    data : ndarray
    requires_grad : bool, default=False
    dtype : default=None

    Attributes
    ----------
    data : numpy.ndarray
    requires_grad : bool
    grad : numpy.ndarray
    next : list[Tensor]
        Nodes that depend on this node.
    last : list[Tensor]
        Nodes that this node depends on.
    """

    def __init__(
        self,
        data: Any,
        dtype=None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data

        self.data = np.array(data, dtype)

        self.requires_grad: bool = requires_grad and is_grad_enable()
        assert not (
            self.requires_grad and self.dtype != float
        ), "Only Tensors of floating point dtype can require gradients!"
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

        self.next: List[Tensor] = list()
        self.last: List[Tensor] = list()

        if self.requires_grad:
            Graph.add_node(self)

    @property
    def is_leaf(self) -> bool:
        return not self.requires_grad or len(self.last) == 0

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def astype(self, new_type):
        assert not self.requires_grad
        self.data.astype(new_type)

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def max(
        self,
        axis: Optional[Union[int, Tuple]] = None,
        keepdims: bool = False,
    ):
        return max(self, axis, keepdims)

    def min(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return min(self, axis, keepdims)

    def mean(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return mean(self, axis, keepdims)

    def sum(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return sum(self, axis, keepdims)

    def build_edge(self, node):
        self.next.append(node)
        node.last.append(self)

    def __repr__(self) -> str:
        return (
            "{}({}, requires_grad={}".format(
                "Tensor",
                self.data,
                self.requires_grad,
            )
            + ")"
        )

    def __add__(self, x):
        return add(self, x)

    def __radd__(self, x):
        return add(x, self)

    def __sub__(self, x):
        return sub(self, x)

    def __rsub__(self, x):
        return sub(x, self)

    def __mul__(self, x):
        return mul(self, x)

    def __rmul__(self, x):
        return mul(x, self)

    def __matmul__(self, x):
        return matmul(self, x)

    def __rmatmul__(self, x):
        return matmul(x, self)

    def __truediv__(self, x):
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def __setitem__(self, key, value):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(key, Tensor):
            key = key.data
        if not isinstance(value, Tensor):
            self.data[key] = value
        else:
            self.data[key] = value.data

    def __len__(self) -> int:
        return len(self.data)

    def __iadd__(self, other):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data += other
        return self

    def __isub__(self, other):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data -= other
        return self

    def __imul__(self, other):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data *= other
        return self

    def __itruediv__(self, other):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data /= other
        return self

    def __imatmul__(self, other):
        assert (
            not self.requires_grad
        ), "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data @= other
        return self

    @no_grad()
    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other)

    @no_grad()
    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other)

    @no_grad()
    def eq(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other)

    @no_grad()
    def ne(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other)

    @no_grad()
    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other)

    @no_grad()
    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other)

    def backward(self, retain_graph: bool = False):
        if self not in Graph.node_list:
            print("AD failed because the node is not in graph.")
            return

        assert self.data.ndim == 0, "backward should be called only on a scalar."

        self.grad = np.ones_like(self.data)
        # Graph.node_list.reverse()
        # y_id = Graph.node_list.index(self)
        # Graph.node_list.reverse()

        y_id = 0
        for i in range(len(Graph.node_list) - 1, -1, -1):
            if Graph.node_list[i] is self:
                y_id = i
                break

        # assert y_id == y_id

        for node in Graph.node_list[y_id::-1]:
            grad = node.grad
            for last in [l for l in node.last if l.requires_grad]:
                add_grad = node.grad_fn(last, grad)
                if add_grad.shape != last.shape:
                    add_grad = np.sum(
                        add_grad,
                        axis=tuple(
                            -i for i in range(1, last.ndim + 1) if last.shape[-i] == 1
                        ),
                        keepdims=True,
                    )
                    add_grad = np.sum(
                        add_grad,
                        axis=tuple(range(add_grad.ndim - last.ndim)),
                    )
                last.grad += add_grad

            if not node.is_leaf:
                node.grad = None

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def item(self):
        return self.data.item()


class UnaryOperator(Tensor):
    """
    Base class for unary operator
    """

    def __init__(self, x) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        super().__init__(
            data=self.forward(x),
            requires_grad=is_grad_enable() and x.requires_grad,
        )
        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        x : Tensor
        grad : ndarray
            Gradient from the last node
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class BinaryOperator(Tensor):
    """
    Base class for binary operator
    """

    def __init__(self, x, y) -> None:
        if not isinstance(x, Tensor) and isinstance(y, Tensor):
            x = Tensor(x)
        elif isinstance(x, Tensor) and not isinstance(y, Tensor):
            y = Tensor(y)
        elif not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            x, y = Tensor(x), Tensor(y)
        super().__init__(
            data=self.forward(x, y),
            requires_grad=is_grad_enable() and (x.requires_grad or y.requires_grad),
        )
        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class add(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return grad[...]


class sub(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return grad[...]
        return -grad


class mul(BinaryOperator):
    """
    Element-wise multiplication
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return self.last[1].data * grad
        return self.last[0].data * grad


class div(BinaryOperator):
    """
    Division
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.last[1].data
        if node is self.last[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    """
    Power
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data**y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * np.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    """
    Matrix multiplication
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            if self.last[1].ndim == 1:
                return np.expand_dims(grad, -1) @ np.expand_dims(self.last[1].data, -2)
            elif self.last[1].ndim > 2:
                shape = list(range(self.last[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.last[1].data.transpose(*shape)
            return grad @ self.last[1].data.T
        else:
            if self.last[0].ndim == 1:
                return np.expand_dims(self.last[0].data, -1) @ np.expand_dims(grad, -2)
            elif self.last[0].ndim > 2:
                shape = list(range(self.last[0].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return self.last[0].data.transpose(*shape) @ grad
            return self.last[0].data.T @ grad


class abs(UnaryOperator):
    """
    Abs
    """

    def forward(self, x: Tensor) -> np.ndarray:
        return np.abs(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        mask = np.zeros(x.shape)
        mask[x > 0] = 1.0
        mask[x < 0] = -1.0
        return grad * mask


class sum(UnaryOperator):
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad


class mean(UnaryOperator):
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad * self.data.size / x.data.size


class max(UnaryOperator):
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # restore dimension
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class min(UnaryOperator):
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # restore dimension
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class exp(UnaryOperator):
    def forward(self, x: Tensor):
        return np.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    """
    Actually, it is ln
    """

    def forward(self, x: Tensor):
        return np.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.minimum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


class transpose(UnaryOperator):
    def __init__(self, x: Tensor, axes: Optional[tuple] = None) -> None:
        self.axes = axes
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class get_slice(UnaryOperator):
    def __init__(self, x: Tensor, key) -> None:
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        full_grad = np.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


def zeros(shape, dtype=None, requires_grad=False):
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=None, requires_grad=False):
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad)


def rand(*shape, dtype=None, requires_grad=False):
    return Tensor(np.random.rand(*shape), dtype=dtype, requires_grad=requires_grad)


def uniform(low: float, high: float, shape=None, dtype=None, requires_grad=False):
    return Tensor(
        np.random.uniform(low, high, size=shape),
        dtype=dtype,
        requires_grad=requires_grad,
    )


def empty(shape, dtype=None, requires_grad=False):
    return Tensor(np.empty(shape), dtype=dtype, requires_grad=requires_grad)
