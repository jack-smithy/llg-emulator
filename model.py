from magnumnp.field_terms import DemagField as _DemagField
from magnumnp.common import Mesh, State
from torch import Tensor
import torch
import torch.nn as nn
from magnumnp.common.constants import mu_0

torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")


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
        nn.init.zeros_(self.cnn[-1].weight)
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


def effective_field(
    model: WallEnergy,
    demag_field: DemagField,
    exchange_field: ExchangeField,
    m: Tensor,
    h_ext: Tensor,
) -> Tensor:
    """h_demag + h_exchange + h_ext + h_theta: m (B, X, Y, Z, 3), h_ext (B, 3) [A/m] -> (B, X, Y, Z, 3) [A/m]."""
    return demag_field(m) + exchange_field(m) + h_ext[:, None, None, None, :] + model.field(m)
