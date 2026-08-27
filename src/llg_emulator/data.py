"""Trajectory loading and the grain step-pair source.

Layout: ``data/<split>/<variant>/sample-XXXXX-of-YYYYY/{m.npy, metadata.json}``.
Every subdirectory of a split is a variant, so new ones are picked up with no
code change. `m.npy` is `(t, nx+1, ny+1, 3)` float32 **nodal** (see physics.py);
`metadata.json` carries the cell counts `n`, the cell size `dx`, the solver step
`dt`, `material.Ms` and the applied field `H_ext` in A/m.

**Trajectories are memory-mapped, not read.** The dataset is 59 GB over 769
trajectories while one training sample touches two frames (~0.8 MB), so the flat
index reads frames on demand and lets the OS page cache do the caching. Peak RSS
for mapping all 769 is ~30 MB.
"""

import json
from pathlib import Path
from typing import NamedTuple

import grain
import numpy as np
from einops import rearrange

# The step size the FiLM step-size input is measured in: s_enc = log2(dt_step /
# BASE_STEP_TIME), so s_enc == 0 is one 10 ps solver step whatever `dt` a future
# variant is generated with.
BASE_STEP_TIME = 10e-12


def load_metadata(path: Path) -> dict:
    """Metadata json for one trajectory"""
    with open(path / "metadata.json") as f:
        return json.load(f)


def _frames_to_cf(arr) -> np.ndarray:
    """(t, h, w, 3) [any dtype] -> contiguous float32 channel-first (t, 3, h, w).

    h/w are the *nodal* grid (mesh cells + 1); see physics.py."""
    arr = rearrange(np.asarray(arr), "t h w c -> t c h w")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _frame_to_cf(arr, i: int) -> np.ndarray:
    """One frame of a memmapped trajectory -> contiguous float32 (3, h, w)."""
    return np.ascontiguousarray(np.moveaxis(arr[i], -1, 0), dtype=np.float32)


class Trajectory(NamedTuple):
    """One memory-mapped trajectory plus the metadata the model conditions on."""

    m: np.ndarray  # memmap (t, nx+1, ny+1, 3), nodal
    H: np.ndarray  # (3,) applied field, nondimensionalised by Ms
    dt: float  # solver step in seconds
    n: tuple  # mesh *cell* counts (nx, ny); nodal grid is n + 1
    dx: tuple  # cell size (dx, dy, thickness)

    def s_enc(self, stride: int) -> np.float32:
        """Step-size input for a `stride`-frame jump, in units of BASE_STEP_TIME."""
        return np.float32(np.log2(stride * self.dt / BASE_STEP_TIME))


def open_trajectory(path: Path) -> Trajectory:
    """Memory-map one trajectory and read its metadata. Does not touch the frames."""
    md = load_metadata(path)
    Ms = np.float32(md["material"]["Ms"])
    return Trajectory(
        m=np.load(path / "m.npy", mmap_mode="r"),
        H=np.asarray(md["H_ext"], dtype=np.float32) / Ms,
        dt=float(md["dt"]),
        n=tuple(int(x) for x in md["n"]),
        dx=tuple(float(x) for x in md["dx"]),
    )


def load_trajectory(path: Path):
    """One *full* trajectory read into memory: contiguous float32 (t, c, h, w) and
    its constant field (3,), nondimensionalised by Ms. For single-trajectory eval
    (SP4/bulk rollout), where every frame is needed anyway."""
    trj = _frames_to_cf(np.load(path / "m.npy"))
    md = load_metadata(path)
    Ms = np.float32(md["material"]["Ms"])
    return trj, np.asarray(md["H_ext"], dtype=np.float32) / Ms


def sample_dirs(dirs) -> list[Path]:
    """All `sample-*` dirs across the given variant dirs, variant-major."""
    return [s for d in dirs for s in sorted(d.glob("sample-*")) if s.is_dir()]


class LLGStepperSource(grain.sources.RandomAccessDataSource):
    """Flat `(trajectory, t, stride)` index over `(m_t, m_{t+s}, m_{t+2s})` triples.

    The third frame is the pushforward target (`training.loss_fn`); the index
    therefore stops two strides short, costing one sample per trajectory.

    All trajectories in one source must share a mesh: grain stacks the samples
    into a batch, and a batch of mixed grids has no shape. Mixed-mesh *training*
    would need one batch stream per mesh (and one demag per mesh); mixed-mesh
    *evaluation* already works, since eval runs one trajectory at a time.
    """

    def __init__(
        self,
        dirs,
        strides=(1,),
        max_trajectories: int | None = None,
    ):
        paths = sample_dirs(dirs)
        if not paths:
            raise FileNotFoundError(f"no sample-* dirs under {[str(d) for d in dirs]}")
        if max_trajectories is not None:
            paths = paths[:max_trajectories]

        self.trajs = [open_trajectory(p) for p in paths]

        meshes = {t.n for t in self.trajs}
        if len(meshes) > 1:
            raise ValueError(
                f"a batched source needs one mesh, got {sorted(meshes)}. Split the "
                f"variants into one source per mesh (see the class docstring)."
            )
        # the mesh this split's demag tensor has to be built for
        self.n, self.dx = self.trajs[0].n, self.trajs[0].dx

        self.index = np.array(
            [
                (ti, t, s)
                for ti, trj in enumerate(self.trajs)
                for s in strides
                for t in range(trj.m.shape[0] - 2 * s)
            ],
            dtype=np.int64,
        )

    @property
    def fields(self) -> list[np.ndarray]:
        return [t.H for t in self.trajs]

    def __getitem__(self, idx: int) -> dict:
        ti, t, s = self.index[idx]
        trj = self.trajs[ti]
        return {
            "m0": _frame_to_cf(trj.m, t),
            "m1": _frame_to_cf(trj.m, t + s),
            "m2": _frame_to_cf(trj.m, t + 2 * s),  # pushforward target
            "H": trj.H,
            "s0": trj.s_enc(int(s)),
        }

    def __len__(self) -> int:
        return len(self.index)


def dataloader_factory(
    source,
    batch_size: int,
    shuffle=True,
    drop_remainder=True,
):
    """`seed -> IterDataset` closure; a fresh shuffle order per epoch."""

    def closure(seed):
        ds = grain.MapDataset.source(source)

        if shuffle:
            ds = ds.shuffle(seed=seed)

        ds = ds.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        ).to_iter_dataset()

        return ds

    return closure
