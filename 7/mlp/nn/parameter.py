from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(
            data=data.data,
            dtype=data.dtype,
            requires_grad=True,
        )

    def __repr__(self) -> str:
        return "Parameter : \n{}".format(self.data)
