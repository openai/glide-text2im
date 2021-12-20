import math
from typing import Callable, Optional

import attr
import torch
import torch.nn as nn
import torch.nn.functional as F

FilterFn = Callable[[torch.Tensor], torch.Tensor]


class ZeroKeyBiasGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, output_grad):
        output_grad = output_grad.clone()
        output_grad.chunk(3)[1].zero_()
        return output_grad


def zero_key_bias_grad(x: torch.Tensor) -> torch.Tensor:
    return ZeroKeyBiasGrad.apply(x)


@attr.s(eq=False, repr=False)
class LayerNorm(nn.Module):
    n_state: int = attr.ib()
    eps: float = attr.ib(default=1e-6)
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones((self.n_state,), dtype=torch.float32, device=self.device))
        self.b = nn.Parameter(torch.zeros((self.n_state,), dtype=torch.float32, device=self.device))
        self.g.weight_decay_level = "disable"  # type: ignore
        self.b.weight_decay_level = "disable"  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.type(torch.float32), torch.Size((self.n_state,)), self.g, self.b, self.eps
        )


@attr.s(eq=False, repr=False)
class Affine(nn.Module):
    n_in: int = attr.ib()
    n_out: int = attr.ib()
    use_bias: bool = attr.ib(default=True)
    use_admnet_init: bool = attr.ib(default=False)
    std: Optional[float] = attr.ib(default=None)
    extra_init_scale: Optional[float] = attr.ib(default=None)
    bias_filter_fn: FilterFn = attr.ib(default=lambda x: x)
    device: torch.device = attr.ib(default=torch.device("cuda"))

    def __attrs_post_init__(self) -> None:
        super().__init__()

        if not self.use_admnet_init:
            self.std = self.std if self.std is not None else math.sqrt(2 / (self.n_in + self.n_out))
            self.std = (
                self.std if self.extra_init_scale is None else self.std * self.extra_init_scale
            )

            w = torch.empty((self.n_out, self.n_in), dtype=torch.float32, device=self.device)
            self.w = nn.Parameter(w)

            if self.use_bias:
                self.b = nn.Parameter(
                    torch.zeros((self.n_out,), dtype=torch.float32, device=self.device)
                )
                self.b.weight_decay_level = "disable"  # type: ignore
        else:
            if self.extra_init_scale is not None:
                raise ValueError("extra_init_scale incompatible with admnet init")

            w = torch.empty((self.n_out, self.n_in), dtype=torch.float32, device=self.device)

            if self.use_bias:
                b = torch.empty((self.n_out,), dtype=torch.float32, device=self.device)

            self.w = nn.Parameter(w)

            if self.use_bias:
                self.b = nn.Parameter(b)
                self.b.weight_decay_level = "disable"  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w if self.w.dtype == x.dtype else self.w.to(x.dtype)
        b = (
            self.bias_filter_fn(self.b if self.b.dtype == x.dtype else self.b.to(x.dtype))
            if self.use_bias
            else None
        )
        return F.linear(x, w, b)
