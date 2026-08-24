import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from magnumnp import (
    DemagField,
    ExchangeField,
    ExternalField,
    LLGSolver,
    Mesh,
    MinimizerBB,
    State,
    set_log_level,
)
from scipy.constants import mu_0
from tqdm import tqdm

set_log_level(25)  # show info_green, but hide info_blue


@dataclass
class SimulationParams:
    n: Sequence[int]
    dx: Sequence[float]
    dt: float
    t_tot: float
    material: dict[str, float]
    H_ext: Sequence[float]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "params.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)


def run(params: SimulationParams, init_m_fn: Callable):
    # initialize state
    mesh = Mesh(params.n, params.dx)
    state = State(mesh)
    state.material = params.material

    # initialize field terms
    demag = DemagField()
    exchange = ExchangeField()

    external = ExternalField(h=params.H_ext)

    init_m_fn(state)

    minimizer = MinimizerBB([demag, exchange])
    minimizer.minimize(state)

    # perform integration with external field
    llg = LLGSolver([demag, exchange, external])
    ms = []
    for _ in tqdm(torch.arange(0, params.t_tot, params.dt), disable=True):
        llg.step(state, params.dt)
        ms.append(state.m.clone())  # type: ignore
    ms = torch.stack(ms, dim=0).detach().cpu().numpy()
    coords = [c.detach().cpu().numpy() for c in mesh.SpatialCoordinate()]

    return ms, coords


def init_m_random(state: State):
    state.m = state.RandM()  # type: ignore


def init_m_s_state(state: State, inv: bool):
    state.m = state.Constant([0, 0, 0])  # type: ignore
    state.m[1:-1, :, :, 0] = -1.0 if inv else 1.0  # type: ignore
    state.m[(-1, 0), :, :, 1] = -1.0 if inv else 1.0  # type: ignore


def init_m_uniform(state: State, m0: Sequence[float]):
    assert len(m0) == 3
    state.m = state.Constant(m0)  # type: ignore


def main():
    # example generation
    params = SimulationParams(
        n=(256, 256, 1),
        dx=(5e-9, 5e-9, 3e-9),
        dt=10e-12,
        t_tot=1e-9,
        material={"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02},
        H_ext=[10e-3 / mu_0, 0, 0],
    )
    ms, coords = run(params=params, init_m_fn=init_m_random)
    # np.save(...) pretend to save them
