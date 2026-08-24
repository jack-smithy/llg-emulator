import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import neuralmag as nm
import numpy as np
from scipy.constants import mu_0
from tqdm.autonotebook import tqdm

nm.config.dtype = "float32"
nm.set_log_level(25)

NM_CACHE = Path(__file__).resolve().parents[2] / ".nm_cache"


@dataclass
class SimulationParams:
    n: Sequence[int]  # mesh cell counts; 2D (nx, ny) for a thin film
    dx: Sequence[float]  # always 3 entries; dx[2] is the film thickness
    dt: float
    t_tot: float
    material: dict[str, float]
    H_ext: Sequence[float]
    init: str  # label of the initial condition, e.g. "s_state_inv"

    def json_dump(self, save_path: Path):
        with (save_path / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def from_json(cls, load_path: Path) -> "SimulationParams":
        """Inverse of `json_dump`; `load_path` is the sample dir, not the file.
        Sequences come back as lists (JSON has no tuples), which `run` accepts."""
        with (Path(load_path) / "metadata.json").open(encoding="utf-8") as file:
            return cls(**json.load(file))


def run(params: SimulationParams, init_m_fn: Callable):
    # initialize state
    mesh = nm.Mesh(tuple(params.n), tuple(params.dx))
    state = nm.State(mesh)
    for name, value in params.material.items():
        setattr(state.material, name, value)

    init_m_fn(state)

    # relax to equilibrium without the external field
    nm.ExchangeField().register(state, "exchange")
    nm.DemagField(cache_dir=NM_CACHE).register(state, "demag")
    nm.TotalField("exchange", "demag").register(state)
    llg = nm.LLGSolver(state)
    llg.relax()

    # perform integration with external field
    nm.ExternalField(state.tensor(params.H_ext)).register(state, "external")
    nm.TotalField("exchange", "demag", "external").register(state)
    llg.reset()
    state.t = 0.0  # relax() integrates in time; restart the clock for the dynamics

    ms = [np.asarray(state.m.tensor)]
    for _ in range(round(params.t_tot / params.dt)):
        llg.step(params.dt)
        ms.append(np.asarray(state.m.tensor))
    ms = np.stack(ms, axis=0)
    # m is nodal, so report node coordinates
    coords = [np.asarray(c) for c in state.coordinates("n" * mesh.dim)]

    return ms, coords


def _nodal_shape(state) -> tuple[int, ...]:
    return tuple(n + 1 for n in state.mesh.n) + (3,)


def _random_unit(key, shape: tuple[int, ...] = ()):
    v = jr.normal(key, shape + (3,))
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def init_m_random(state, key):
    m = _random_unit(key, _nodal_shape(state)[:-1])
    state.m = nm.VectorFunction(state, tensor=state.tensor(m))


def init_m_s_state(state, inv: bool):
    m = np.zeros(_nodal_shape(state))
    m[1:-1, ..., 0] = -1.0 if inv else 1.0
    m[(-1, 0), ..., 1] = -1.0 if inv else 1.0
    state.m = nm.VectorFunction(state, tensor=state.tensor(m))


def init_m_uniform(state, m0: Sequence[float]):
    assert len(m0) == 3
    state.m = nm.VectorFunction(state).fill(tuple(map(float, m0)))


def random_init(key) -> tuple[str, Callable]:
    """Pick one of the initial conditions at random. Returns `(label, fn)` with
    `fn` a `state -> None` callable, so `run` stays unaware of the choice and the
    label can go into the metadata."""
    k_choice, k = jr.split(key)
    choice = int(jr.randint(k_choice, (), 0, 3))
    if choice == 0:
        return "random", lambda state: init_m_random(state, k)
    if choice == 1:
        inv = bool(jr.bernoulli(k))
        return f"s_state{'_inv' if inv else ''}", lambda state: init_m_s_state(
            state, inv=inv
        )
    m0 = _random_unit(k)
    return (
        f"uniform{tuple(round(float(x), 4) for x in m0)}",
        lambda state: init_m_uniform(state, m0),
    )


def random_H_ext(key, max_field: float = 30e-3 / mu_0) -> list[float]:
    """In-plane applied field, uniform in [-max_field, max_field]^2."""
    return jr.uniform(key, (3,), minval=-max_field, maxval=max_field).tolist()


def generate_sample(
    seed: int, n_samples: int, index: int, base_dir: Path, n, dx, t_tot
) -> None:
    """Generate and save sample `index` (1-based). Its key depends only on
    `(seed, n_samples, index)`, so workers need no shared state and how the
    samples are sharded across GPUs cannot change the output."""
    k_H, k_init = jr.split(jr.split(jr.PRNGKey(seed), n_samples)[index - 1])
    init_label, init_m_fn = random_init(k_init)

    params = SimulationParams(
        n=n,
        dx=dx,
        dt=10e-12,
        t_tot=t_tot,
        material={"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02},
        H_ext=random_H_ext(k_H),
        init=init_label,
    )

    ms, _ = run(params=params, init_m_fn=init_m_fn)

    save_dir = base_dir / f"sample_{index}"
    save_dir.mkdir(parents=True, exist_ok=False)

    params.json_dump(save_path=save_dir)
    np.save(save_dir / "m.npy", ms)


def main(
    seed: int = 0,
    n_samples: int = 4,
    base_dir: Path = Path("data/train"),
    n: Sequence[int] = (255, 255, 1),
    dx: Sequence[float] = (5e-9, 5e-9, 3e-9),
    t_tot: float = 1e-9,
):
    """Generate `n_samples` trajectories into `base_dir`."""
    for index in tqdm(range(1, n_samples + 1)):
        generate_sample(seed, n_samples, index, base_dir, n, dx, t_tot)


if __name__ == "__main__":
    main(n_samples=512)
