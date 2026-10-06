"""magnum.np, the reference solver the datasets were generated with (torch)."""

from collections.abc import Iterator

import numpy as np
import torch
from jaxtyping import Float
from magnumnp import (
    DemagField,
    ExchangeField,
    ExternalField,
    LLGSolver,
    Mesh,
    State,
    normalize,
    set_log_level,
)

from llg.constants import DT, MATERIAL

set_log_level(250)

Field = Float[np.ndarray, "..."]


def random_state(n: tuple[int, int], gen: torch.Generator) -> torch.Tensor:
    """Per-cell random directions: relaxes into a multi-domain or vortex state."""
    return random_directions(n + (1,), gen)


def uniform_state(n: tuple[int, int], gen: torch.Generator) -> torch.Tensor:
    return torch.zeros(n + (1, 3)) + random_directions((), gen)


def s_state(n: tuple[int, int], gen: torch.Generator | None = None) -> torch.Tensor:
    """SP4's initial condition, +x with the ends canted; the cant randomised by gen."""
    cant, sign, width = 1.0, 1.0, 1
    if gen is not None:
        cant = 0.2 + 0.8 * torch.rand((), generator=gen)
        sign = 1.0 if torch.rand((), generator=gen) < 0.5 else -1.0
        width = int(torch.randint(1, 5, (), generator=gen))
    m = torch.zeros(n + (1, 3))
    m[..., 0] = 1.0
    m[:width, :, :, 1] = sign * cant
    m[-width:, :, :, 1] = -sign * cant
    return m


def random_directions(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    theta = 2.0 * torch.pi * torch.rand(shape, generator=gen)
    phi = torch.acos(2.0 * torch.rand(shape, generator=gen) - 1.0)
    return torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)],
        dim=-1,
    )


def relax(m0: torch.Tensor, dx: tuple[float, ...]) -> Field:
    """Relaxed at zero field (magnum.np's relax runs at alpha = 1)."""
    state = make_state(m0, dx)
    LLGSolver([DemagField(), ExchangeField()]).relax(state)
    return to_numpy(normalize(state.m))  # type: ignore


def simulate(
    m0: Field | torch.Tensor, h: Float[np.ndarray, "..."], dx: tuple[float, ...], n_steps: int
) -> Iterator[Field]:
    """The n_steps frames after m0 under a constant applied field h in A/m."""
    state = make_state(m0, dx)
    llg = LLGSolver([DemagField(), ExchangeField(), ExternalField([float(x) for x in h])])
    for _ in range(n_steps):
        llg.step(state, DT)
        yield to_numpy(state.m)  # type: ignore


def make_state(m0: Field | torch.Tensor, dx: tuple[float, ...]) -> State:
    m0 = torch.as_tensor(m0, dtype=torch.get_default_dtype())
    if m0.ndim == 3:
        m0 = m0[:, :, None]
    state = State(Mesh(tuple(m0.shape[:3]), dx))
    state.material = dict(MATERIAL)
    state.m = normalize(m0.to(torch.get_default_device()).clone())  # type: ignore
    return state


def to_numpy(m: torch.Tensor) -> Field:
    return m.squeeze(2).cpu().numpy()
