import json
from pathlib import Path

import grain
import numpy as np
from einops import rearrange

from llg_emulator.train_config import TrainConfig


def load_metadata(path: Path) -> dict:
    """Metadata json for one trajectory"""
    with open(path / "params.json") as f:
        return json.load(f)


def _frames_to_cf(arr) -> np.ndarray:
    """(t, h, w, 1, 3) [any dtype] -> contiguous float32 channel-first (t, 3, h, w)."""
    arr = rearrange(np.asarray(arr).squeeze(-2), "t h w c -> t c h w")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_trajectory(path: Path):
    """One *full* trajectory: contiguous float32 (t, c, h, w), its constant field
    (3,), and the base solver timestep dt (s). Reads the whole m.npy — fine for
    single-trajectory eval (SP4/bulk rollout), NOT for the training sources, which
    stream (see TrajectoryStore)."""
    trj = _frames_to_cf(np.load(path / "m.npy"))
    params = load_metadata(path)
    Ms = np.float32(params["material"]["Ms"])
    H = np.asarray(params["H_ext"], dtype=np.float32) / Ms
    dt = float(params["dt"])
    return trj, H, dt


class TrajectoryStore:
    """Lazily-streamed access to a directory of trajectories.

    At construction it reads only each `params.json` (for H, dt) and the `m.npy`
    header (for the frame count) — **no frame data**. Frames are memory-mapped and
    sliced on demand, so host memory stays O(index + a few live frames) regardless
    of dataset size. This is what lets training scale to datasets far larger than
    RAM (the previous sources loaded every m.npy fully at init and OOM'd).

    mmap handles are cached per file. The grain pipeline here is thread-based
    (device_put/thread-prefetch, not multiprocessing), so a shared handle cache is
    safe — numpy mmap reads are concurrent-read safe.
    """

    def __init__(self, path: Path):
        self.dirs = sorted(p for p in Path(path).iterdir() if p.is_dir())
        self.fields, self.dts, self.lengths = [], [], []
        for d in self.dirs:
            params = load_metadata(d)
            Ms = np.float32(params["material"]["Ms"])
            self.fields.append(np.asarray(params["H_ext"], dtype=np.float32) / Ms)
            self.dts.append(float(params["dt"]))
            # header-only read (~0.3 ms): frame count without loading data
            self.lengths.append(int(np.load(d / "m.npy", mmap_mode="r").shape[0]))
        self._mmaps: dict[int, np.ndarray] = {}

    def _mmap(self, ti: int) -> np.ndarray:
        mm = self._mmaps.get(ti)
        if mm is None:
            mm = np.load(self.dirs[ti] / "m.npy", mmap_mode="r")
            self._mmaps[ti] = mm
        return mm

    def frames(self, ti: int, sl: slice) -> np.ndarray:
        """Channel-first float32 frames trj[sl] for trajectory ti (reads only sl)."""
        return _frames_to_cf(self._mmap(ti)[sl])

    def frame(self, ti: int, t: int) -> np.ndarray:
        """Single channel-first float32 frame trj[t]."""
        return self.frames(ti, slice(t, t + 1))[0]

    def full(self, ti: int) -> np.ndarray:
        """Whole trajectory (channel-first float32) — for per-trajectory eval."""
        return self.frames(ti, slice(None))

    def __len__(self) -> int:
        return len(self.dirs)


class LLGStepperSource(grain.sources.RandomAccessDataSource):
    """Streams consecutive (m_t, m_{t+1}, H, s_enc=0) one-step pairs (used for val)."""

    def __init__(self, path: Path, max_workers: int = 16):
        self.store = TrajectoryStore(path)
        # small index only: one (traj, t) per consecutive pair across trajectories
        self.index = np.array(
            [
                (ti, t)
                for ti, n in enumerate(self.store.lengths)
                for t in range(n - 1)
            ],
            dtype=np.int64,
        )

    def __getitem__(self, idx: int) -> dict:
        ti, t = (int(x) for x in self.index[idx])
        pair = self.store.frames(ti, slice(t, t + 2))  # (2, 3, nx, ny), one read
        # one-step pairs -> stride 1 -> s_enc = log2(1) = 0
        return {
            "m0": pair[0],
            "m1": pair[1],
            "H": self.store.fields[ti],
            "s_enc": np.float32(0.0),
        }

    def __len__(self) -> int:
        return len(self.index)


def dataloader_factory(
    source,
    device,
    config: TrainConfig,
    shuffle=True,
    drop_remainder=True,
):
    def closure(seed):
        ds = grain.MapDataset.source(source)

        if shuffle:
            ds = ds.shuffle(seed=seed)

        ds = ds.batch(
            batch_size=config.batch_size,
            drop_remainder=drop_remainder,
        ).to_iter_dataset()

        ds = grain.experimental.device_put(
            ds=ds,
            device=device,
            cpu_buffer_size=config.cpu_buffer_size,  # batches buffered on host
            device_buffer_size=config.device_buffer_size,  # batches buffered on device
        )
        return ds

    return closure
