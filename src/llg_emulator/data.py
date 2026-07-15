import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grain
import numpy as np
from einops import rearrange
from tqdm import tqdm

from llg_emulator.train_config import TrainConfig

BASE_STEP_TIME = 10e-12  # 10ps


def load_metadata(path: Path) -> dict:
    """Metadata json for one trajectory"""
    with open(path / "params.json") as f:
        return json.load(f)


def load_trajectory(path: Path):
    """One trajectory as contiguous float32 (t, c, h, w) + its constant field (3,)."""
    trj = np.load(path / "m.npy")  # mmap buys nothing; it's read in full
    trj = rearrange(trj.squeeze(-2), "t h w c -> t c h w")
    trj = np.ascontiguousarray(trj, dtype=np.float32)

    params = load_metadata(path)
    Ms = np.float32(params["material"]["Ms"])
    H = np.asarray(params["H_ext"], dtype=np.float32) / Ms
    return trj, H


def load_trajectories(
    path: Path,
    num_shards: int | None,
    max_workers: int = 16,
):
    """Read all trajectories in parallel from disk"""
    dirs = sorted(p for p in path.iterdir() if p.is_dir())

    if num_shards is not None:
        if num_shards > len(dirs):
            raise ValueError(f"shards selected={num_shards}, total shards={len(dirs)}")
        dirs = dirs[:num_shards] if num_shards is not None else dirs

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(load_trajectory, dirs), total=len(dirs)))
    trajs, fields = zip(*results)
    return list(trajs), list(fields)


class LLGStepperSource(grain.sources.RandomAccessDataSource):
    def __init__(
        self,
        path: Path,
        max_workers: int = 16,
        strides=[1, 2, 4, 8],
        num_shards: int | None = None,
    ):
        self.trajs, self.fields = load_trajectories(path, num_shards, max_workers)
        # flat index: one (traj, t) entry per consecutive pair, across all trajectories
        self.index = np.array(
            [
                (ti, t, s)
                for ti, trj in enumerate(self.trajs)
                for s in strides
                for t in range(trj.shape[0] - s)
            ],
            dtype=np.int64,
        )

    def __getitem__(self, idx: int) -> dict:
        ti, t, s = self.index[idx]
        trj = self.trajs[ti]
        s0 = np.log2(s)
        return {"m0": trj[t], "m1": trj[t + s], "H": self.fields[ti], "s0": s0}

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

        return ds

    return closure
