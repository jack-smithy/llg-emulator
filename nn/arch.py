from .layers import (
    Activation,
    BoundaryMode,
    PointwiseLinearConv3d,
    residual_constructor_3d,
)


from torch import Tensor, nn


class ResNet3d(nn.Module):
    """Pointwise lift, residual blocks of two 3x3x3 convolutions with zero (AIRBOX) padding, pointwise
    projection; the mask zeroes padded cells inside every block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_hidden_layers: int,
        kernel_size: int,
        activation: Activation,
        use_norm: bool,
        zero_projection_layer: bool,
    ) -> None:
        super().__init__()
        self.lifting = PointwiseLinearConv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            bias=False,
        )

        residuals = residual_constructor_3d(
            kernel_size=kernel_size,
            use_norm=use_norm,
            activation=activation,
            boundary_mode=BoundaryMode.AIRBOX,
        )
        self.hidden = nn.ModuleList(
            [
                residuals(hidden_channels, hidden_channels)
                for _ in range(num_hidden_layers)
            ]
        )

        self.projection = PointwiseLinearConv3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=False,
        )
        if zero_projection_layer:
            # the energy starts at 0: training starts from coarse physics
            nn.init.zeros_(self.projection.weight)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.lifting(x)  # no bias: padded cells stay 0
        for block in self.hidden:
            x = block(x, mask)
        return self.projection(x)
