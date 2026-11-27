import itertools

from magnumnp.field_terms import DemagField as _DemagField
from magnumnp.common import Mesh, State
from torch import Tensor
import torch
import torch.nn as nn
from magnumnp.common.constants import mu_0
from nn.layers import Activation, residual_constructor_3d, PointwiseLinearConv3d


class EnergyModel(nn.Module):
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
        )

        self.hidden = nn.Sequential(
            *[
                residuals(hidden_channels, hidden_channels)
                for _ in range(num_hidden_layers)
            ]
        )

        self.projection = PointwiseLinearConv3d(
            in_channels=hidden_channels, out_channels=out_channels, bias=False
        )
        # E_theta starts at 0, so training starts from exchange + demag + applied field: with no
        # exchange-like stiffness the 5 nm minimisation does not converge
        if zero_projection_layer:
            nn.init.zeros_(self.projection.weight)

    def forward(self, x):
        x = self.lifting(x)
        x = self.hidden(x)
        return self.projection(x)


class DemagField(nn.Module):
    def __init__(self, n, dx, Ms, device):
        super().__init__()
        _kernel = _DemagField()._init_N(State(Mesh(n=n, dx=dx)))
        self.kernel = [[component.to(device) for component in row] for row in _kernel]
        self.Ms = Ms

    def forward(self, m: Tensor) -> Tensor:
        """magnum.np's DemagField.h with a batch dimension: m (B, X, Y, Z, 3) -> h (B, X, Y, Z, 3) [A/m]."""
        nx, ny, nz = m.shape[1:4]
        padded = (2 * nx, 2 * ny, 2 * nz)  # zero padding: open boundaries
        m_fft = [
            torch.fft.rfftn(self.Ms * m[..., i], s=padded, dim=(1, 2, 3))
            for i in range(3)
        ]
        h = []
        for row in self.kernel:
            h_fft = row[0] * m_fft[0] + row[1] * m_fft[1] + row[2] * m_fft[2]
            h.append(torch.fft.irfftn(h_fft, s=padded, dim=(1, 2, 3))[:, :nx, :ny, :nz])
        return torch.stack(h, dim=-1)


class ExchangeField(nn.Module):
    def __init__(self, dx, A, Ms):
        super().__init__()
        self.dx = dx
        self.A = A
        self.Ms = Ms

    def forward(self, m: Tensor) -> Tensor:
        """magnum.np's ExchangeField.h for uniform A and Ms, open boundaries, with a batch dimension:
        m (B, X, Y, Z, 3) -> h (B, X, Y, Z, 3) [A/m]. A missing neighbour at the surface adds nothing."""
        dx, dy, dz = self.dx
        h = torch.zeros_like(m)
        diff = (m[:, 1:] - m[:, :-1]) / dx**2
        h[:, :-1] += diff
        h[:, 1:] -= diff
        diff = (m[:, :, 1:] - m[:, :, :-1]) / dy**2
        h[:, :, :-1] += diff
        h[:, :, 1:] -= diff
        diff = (m[:, :, :, 1:] - m[:, :, :, :-1]) / dz**2
        h[:, :, :, :-1] += diff
        h[:, :, :, 1:] -= diff
        return 2 * self.A / (mu_0 * self.Ms) * h


class WallEnergy(nn.Module):
    """E_theta[m] = Kd V sum_cells CNN(m), a learned wall energy added to exchange."""

    def __init__(self, Kd, V, Ms):
        super().__init__()
        # local 3x3x3 stencils like exchange's; SiLU because training differentiates the field
        # h = -dE/dm once more (ReLU has no second derivative); replicate padding: no change across
        # the surface, exchange's free-surface condition
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            # no bias: a constant energy has no field and would never get a gradient
            nn.Conv3d(32, 1, kernel_size=1, bias=False),
        )
        # E_theta starts at 0, so training starts from exchange + demag + applied field: with no
        # exchange-like stiffness the 5 nm minimisation does not converge
        nn.init.zeros_(self.cnn[-1].weight)  # type: ignore
        self.Kd = Kd
        self.V = V
        self.Ms = Ms

    def energy(self, m: Tensor) -> Tensor:
        """m (B, X, Y, Z, 3) -> E (B,) [J]."""
        density = self.cnn(m.permute(0, 4, 1, 2, 3))  # (B, 1, X, Y, Z), in units of Kd
        return self.Kd * self.V * density.sum(dim=(1, 2, 3, 4))

    def field(self, m: Tensor) -> Tensor:
        """h_theta = -dE/dm / (mu0 Ms V): m (B, X, Y, Z, 3) -> (B, X, Y, Z, 3) [A/m]."""
        # the field keeps its graph only when gradients are on, i.e. when training through relax()
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            if not m.requires_grad:
                m = m.detach().requires_grad_()
            (dE_dm,) = torch.autograd.grad(
                self.energy(m).sum(), m, create_graph=create_graph
            )
        return -dE_dm / (mu_0 * self.Ms * self.V)


class WallEnergyResidual(WallEnergy):
    """WallEnergy with EnergyModel as the energy density: pointwise lift, residual blocks of two 3x3x3
    convolutions, pointwise projection (zero at the start, so training starts from coarse physics)."""

    def __init__(self, Kd, V, Ms):
        super().__init__(Kd, V, Ms)
        self.cnn = EnergyModel(
            in_channels=3,
            hidden_channels=32,
            out_channels=1,
            num_hidden_layers=2,
            kernel_size=3,
            use_norm=False,  # GroupNorm over all cells would make the energy non-local
            activation=nn.GELU(),  # smooth: training differentiates h = -dE/dm once more
            zero_projection_layer=True,
        )


class SymmetricWallEnergy(WallEnergy):
    """WallEnergy averaged over the symmetries of a cubic grid without anisotropy: the 48 rotations and
    reflections of the cells (with m transformed as a polar vector, i.e. a reflection combined with
    time reversal) and m -> -m, so E_theta is exactly invariant under all 96."""

    def energy(self, m: Tensor) -> Tensor:
        """m (B, X, Y, Z, 3) -> E (B,) [J], the mean of WallEnergy's energy over the 96 elements.

        One CNN call per axis permutation (the grid shape depends on it), with its 8 flips and 2 signs
        stacked along the batch."""
        energies = []
        for perm in itertools.permutations(range(3)):
            # new axis i is old axis perm[i], and so is the new component i
            g_m = m.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2], 4)[
                ..., list(perm)
            ]
            variants = []
            for flips in itertools.product([False, True], repeat=3):
                flipped = [1 + i for i in range(3) if flips[i]]
                v = g_m.flip(flipped) if flipped else g_m
                component_signs = torch.tensor(
                    [-1.0 if f else 1.0 for f in flips], device=m.device, dtype=m.dtype
                )
                variants += [component_signs * v, -component_signs * v]
            energy = super().energy(torch.cat(variants))  # (16 B,)
            energies.append(energy.view(16, -1))
        return torch.cat(energies).mean(0)


class ContextWallEnergy(WallEnergy):
    """WallEnergy whose energy density also sees the demag field of m and the applied field:
    E_theta[m; h_ext] = Kd V sum_cells CNN(m, h_demag(m) / Ms, h_ext / Ms). Still an energy of m at each
    applied field (h_demag is linear in m, and h_theta = -dE/dm differentiates through it); the applied
    field enters because the sub-cell structure, and so the coarse energy, changes with it. Set h_ext
    (B, 3) [A/m] before calling energy or field (effective_field does)."""

    def __init__(self, Kd, V, Ms, demag_field):
        super().__init__(Kd, V, Ms)
        self.cnn[0] = nn.Conv3d(
            9, 32, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.demag_field = demag_field
        self.h_ext = None

    def energy(self, m: Tensor) -> Tensor:
        h_demag = self.demag_field(m) / self.Ms
        h_ext = (self.h_ext / self.Ms)[:, None, None, None, :].expand_as(m)
        x = torch.cat([m, h_demag, h_ext], dim=-1).permute(0, 4, 1, 2, 3)
        return self.Kd * self.V * self.cnn(x).sum(dim=(1, 2, 3, 4))


def effective_field(
    model: WallEnergy,
    demag_field: DemagField,
    exchange_field: ExchangeField,
    m: Tensor,
    h_ext: Tensor,
) -> Tensor:
    """h_demag + h_exchange + h_ext + h_theta: m (B, X, Y, Z, 3), h_ext (B, 3) [A/m] -> (B, X, Y, Z, 3) [A/m]."""
    if hasattr(model, "h_ext"):  # ContextWallEnergy
        model.h_ext = h_ext
    return (
        demag_field(m)
        + exchange_field(m)
        + h_ext[:, None, None, None, :]
        + model.field(m)
    )
