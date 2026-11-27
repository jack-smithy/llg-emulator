from collections.abc import Callable
from typing import Literal

import torch.nn as nn
from torch import Tensor
from enum import Enum

type Activation = Callable[[Tensor], Tensor]  # must have a second derivative

type PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]


class BoundaryMode(Enum):
    AIRBOX = "airbox"  # vacuum outside: zero ghost cells, as DIRICHLET (m = 0 outside the magnet)
    NEUMANN = "neumann"  # zero normal derivative on a cell-centred grid: ghost cells copy the edge
    DIRICHLET = "dirichlet"  # homogeneous: zero ghost cells
    PERIODIC = "periodic"


PADDING_MODES: dict[BoundaryMode, PaddingMode] = {
    BoundaryMode.AIRBOX: "zeros",
    BoundaryMode.NEUMANN: "replicate",
    BoundaryMode.DIRICHLET: "zeros",
    BoundaryMode.PERIODIC: "circular",
}


def _padding(
    boundary_mode: BoundaryMode, kernel_size: int, dilation: int | tuple[int, int, int]
) -> tuple[PaddingMode, int | tuple[int, int, int]]:
    """Padding mode and width that keep the grid size (at stride 1): dilation * (kernel_size // 2)."""
    assert kernel_size % 2 == 1, "odd kernel sizes only: even ones need asymmetric padding"
    if isinstance(dilation, int):
        padding = dilation * (kernel_size // 2)
    else:
        half = kernel_size // 2
        padding = (dilation[0] * half, dilation[1] * half, dilation[2] * half)
    return PADDING_MODES[boundary_mode], padding


class PhysicsConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        zero_bias_init: bool = False,
        boundary_mode: BoundaryMode = BoundaryMode.NEUMANN,
        device=None,
        dtype=None,
    ) -> None:
        """
        Thin wrapper around torch's Conv3d which handles BCs in the padding
        """
        padding_mode, padding = _padding(boundary_mode, kernel_size, dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        if bias and zero_bias_init:
            assert self.bias is not None
            nn.init.zeros_(self.bias)


class PointwiseLinearConv3d(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, stride: int = 1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            dilation=1,
            padding=0,
            bias=bias,
        )


class ResidualLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        zero_bias_init: bool = False,
        boundary_mode: BoundaryMode = BoundaryMode.NEUMANN,
        use_norm=False,
        activation: Activation = nn.functional.gelu,
    ):
        super().__init__()
        self.conv1 = PhysicsConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            zero_bias_init=zero_bias_init,
            boundary_mode=boundary_mode,
        )
        self.conv2 = PhysicsConv3d(  # stride 1: conv1 alone downsamples, as the bypass does
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            bias=bias,
            zero_bias_init=zero_bias_init,
            boundary_mode=boundary_mode,
        )

        if use_norm:
            self.norm1 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if in_channels == out_channels and stride == 1:
            self.bypass_conv = nn.Identity()
        else:  # match the channels and the downsampling of the main path
            self.bypass_conv = PointwiseLinearConv3d(
                in_channels=in_channels, out_channels=out_channels, bias=False, stride=stride
            )

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        skip = self.bypass_conv(skip)
        return self.activation(x + skip)


def residual_constructor_3d(
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    zero_bias_init: bool = False,
    boundary_mode: BoundaryMode = BoundaryMode.NEUMANN,
    use_norm: bool = False,
    activation: Activation = nn.functional.gelu,
) -> Callable[[int, int], ResidualLayer3d]:
    return lambda i, o: ResidualLayer3d(
        in_channels=i,
        out_channels=o,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=bias,
        zero_bias_init=zero_bias_init,
        boundary_mode=boundary_mode,
        use_norm=use_norm,
        activation=activation,
    )
