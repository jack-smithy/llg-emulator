import json
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import neuralmag as nm
import numpy as np
from jaxtyping import PRNGKeyArray
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
    key: list[int]

    def json_dump(self, save_path: Path):
        with (save_path / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def from_json(cls, load_path: Path) -> "SimulationParams":
        """Inverse of `json_dump`; `load_path` is the sample dir, not the file.
        Sequences come back as lists (JSON has no tuples), which `run` accepts."""
        with (Path(load_path) / "metadata.json").open(encoding="utf-8") as file:
            return cls(**json.load(file))


def run(params: SimulationParams, init_m_fn: Callable, out_path: Path | None = None):
    """Relax, then integrate `t_tot` in `dt` steps, returning `(frames, coords)`.

    With `out_path`, frames are written straight into a memory-mapped `.npy` as
    they are produced and `None` comes back in place of the array. That is what
    makes the big meshes reachable: at 10000x10000 one nodal frame is 1.2 GB, so
    collecting 101 of them in a list and `np.stack`-ing (which copies) needs
    ~242 GB of host RAM. Streaming holds one frame.
    """
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

    n_frames = round(params.t_tot / params.dt) + 1
    shape = (n_frames, *(int(x) + 1 for x in params.n), 3)  # m is nodal
    if out_path is None:
        ms = np.empty(shape, dtype=np.float32)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ms = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=shape
        )

    ms[0] = np.asarray(state.m.tensor)
    for i in range(1, n_frames):
        llg.step(params.dt)
        ms[i] = np.asarray(state.m.tensor)

    # m is nodal, so report node coordinates
    coords = [np.asarray(c) for c in state.coordinates("n" * mesh.dim)]

    if out_path is None:
        return ms, coords
    ms.flush()
    del ms
    return None, coords


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
    key: PRNGKeyArray,
    n_samples: int,
    index: int,
    base_dir: Path,
    n,
    dx,
    t_tot,
    dt,
) -> None:
    """Generate and save sample `index` (1-based). Its key depends only on
    `(seed, n_samples, index)`, so workers need no shared state and how the
    samples are sharded across GPUs cannot change the output."""
    key_h, key_init = jr.split(key)

    init_label, init_m_fn = random_init(key_init)

    params = SimulationParams(
        n=n,
        dx=dx,
        dt=dt,
        t_tot=t_tot,
        material={"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02},
        H_ext=random_H_ext(key_h),
        init=init_label,
        key=key.tolist(),
    )

    save_dir = base_dir / f"sample-{index:05d}-of-{n_samples:05d}"
    save_dir.mkdir(parents=True, exist_ok=False)

    run(params=params, init_m_fn=init_m_fn, out_path=save_dir / "m.npy")
    params.json_dump(save_path=save_dir)


def generate_sp4(
    key: PRNGKeyArray,
    n_samples: int,
    index: int,
    base_dir: Path,
    n,
    dx,
    t_tot,
) -> None:
    params = SimulationParams(
        n=n,
        dx=dx,
        dt=10e-12,
        t_tot=t_tot,
        material={"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02},
        H_ext=[-24.6e-3 / mu_0, 4.3e-3 / mu_0, 0.0],
        init="s_state",
        key=key.tolist(),
    )

    save_dir = base_dir / f"sample-{index:05d}-of-{n_samples:05d}"
    save_dir.mkdir(parents=True, exist_ok=False)

    run(
        params=params,
        init_m_fn=lambda c: init_m_s_state(c, inv=False),
        out_path=save_dir / "m.npy",
    )
    params.json_dump(save_path=save_dir)


def main_fixed_geo():

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)

    base_dir: Path = Path("data")
    n: Sequence[int] = (2000, 2000)
    dx: Sequence[float] = (5e-9, 5e-9, 3e-9)
    t_tot: float = 1e-9
    dt: float = 10e-12

    save_dir = base_dir / args.split / "xlarge"

    keys = jr.split(key, args.n_samples)
    for index in tqdm(range(args.start, args.n_samples + args.start)):
        generate_sample(keys[index], args.n_samples, index, save_dir, n, dx, t_tot, dt)


def main_variable_geo():

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)

    base_dir: Path = Path("data")
    dx: Sequence[float] = (5e-9, 5e-9, 3e-9)
    t_tot: float = 1e-9
    dt: float = 10e-12

    save_dir = base_dir / args.split / "fixed_geo"

    keys = jr.split(key, args.n_samples)
    for index in tqdm(range(args.start, args.n_samples + args.start)):
        sample_key, geo_key = jr.split(keys[index])
        n = jr.randint(geo_key, (2,), minval=16, maxval=512, dtype=jnp.int16).tolist()
        generate_sample(sample_key, args.n_samples, index, save_dir, n, dx, t_tot, dt)


def main_sp4():
    """SP4's field and s-state init on a large mesh -> data/val/<name>.

    Size ceiling on one GPU (RTX PRO 6000 Blackwell, 96 GB), measured 2026-08-28:
    the *solver* is not the limit -- 5000x5000 peaks at 14.4 GB of device memory.
    The surrogate is: a rollout step peaks at 16.2 GB on 3001^2 and 30.2 GB on
    4001^2 (quadratic), and 5001^2 dies on a single 36 GiB allocation even with
    XLA_PYTHON_CLIENT_MEM_FRACTION=.95. So 4000 is the largest mesh that can be
    both simulated and evaluated; going beyond it needs tiled inference.
    """
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=4000, help="cells per side")
    parser.add_argument("--name", type=str, default="sp4_xlarge")
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()

    generate_sp4(
        key=jr.PRNGKey(args.seed),
        n_samples=1,
        index=1,
        base_dir=Path("data") / "val" / args.name,
        n=(args.n, args.n),
        dx=(5e-9, 5e-9, 3e-9),
        t_tot=1e-9,
    )


if __name__ == "__main__":
    main_sp4()
