"""The learned correction: a wall energy E_theta[m] = Kd V sum_cells mask * rho_theta(m), added to coarse
exchange, with field h_theta = -dE_theta/dm / (mu0 Ms V).

Outside the magnet is air: zero padding, and in padded grids (data.StoredStates) the padded cells are
zeroed after every layer and left out of the sum. A magnet padded into a larger grid therefore has
exactly the energy it has alone on its own grid. mask is (B, X, Y, Z), or None for no padding.
"""

import torch
from magnumnp.common.constants import mu_0
from torch import Tensor, nn
from torch.nn import functional as F

from nn.arch import ResNet3d
from nn.layers import Activation


class WallEnergy(nn.Module):
    """The energy density is a ResNet3d: a pointwise lift, residual blocks of two 3x3x3 convolutions
    (local stencils like exchange's; with the default 2 blocks each cell's density sees 4 cells in every
    direction) and a pointwise projection, zero at the start so training starts from coarse physics.
    GELU, a smooth activation, because training differentiates the field h = -dE/dm once more (ReLU has
    no second derivative)."""

    def __init__(
        self,
        Kd: float,
        V: float,
        Ms: float,
        hidden_channels: int = 32,
        hidden_layers: int = 2,
        kernel_size: int = 3,
        activation: Activation = F.gelu,
    ):
        super().__init__()
        self.cnn = ResNet3d(
            in_channels=3,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_hidden_layers=hidden_layers,
            kernel_size=kernel_size,
            activation=activation,  # smooth: training differentiates h = -dE/dm once more
            use_norm=False,  # GroupNorm over all cells would make the energy non-local
            zero_projection_layer=True,
        )
        self.Kd = Kd
        self.V = V
        self.Ms = Ms

    def density(self, m: Tensor, mask: Tensor | None) -> Tensor:
        """Energy density in units of Kd (dimensionless): m (B, X, Y, Z, 3) -> (B, X, Y, Z)."""
        u = self.cnn(m.permute(0, 4, 1, 2, 3), None if mask is None else mask[:, None])
        return u[:, 0]

    def energy(self, m: Tensor, mask: Tensor | None = None) -> Tensor:
        """m (B, X, Y, Z, 3) -> E (B,) [J]."""
        density = self.density(m, mask)
        if mask is not None:
            density = density * mask
        return self.Kd * self.V * density.sum(dim=(1, 2, 3))

    def field(self, m: Tensor, mask: Tensor | None = None) -> Tensor:
        """h_theta = -dE/dm / (mu0 Ms V): m (B, X, Y, Z, 3) -> (B, X, Y, Z, 3) [A/m]."""
        # the field keeps its graph only when gradients are on, i.e. when training
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if not m.requires_grad:
                m = m.detach().requires_grad_()

            (dE_dm,) = torch.autograd.grad(
                outputs=self.energy(m, mask).sum(),
                inputs=m,
                create_graph=create_graph,
            )

        return -dE_dm / (mu_0 * self.Ms * self.V)
