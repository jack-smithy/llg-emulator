import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grain
import numpy as np
from einops import rearrange
from tqdm import tqdm


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


def load_trajectories(path: Path, max_workers: int = 16):
    """Read all trajectories in parallel from disk"""
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(load_trajectory, dirs), total=len(dirs)))
    trajs, fields = zip(*results)
    return list(trajs), list(fields)


class LLGStepperSource(grain.sources.RandomAccessDataSource):
    def __init__(self, path: Path, max_workers: int = 16):
        self.trajs, self.fields = load_trajectories(path, max_workers)
        # flat index: one (traj, t) entry per consecutive pair, across all trajectories
        self.index = np.array(
            [
                (ti, t)
                for ti, trj in enumerate(self.trajs)
                for t in range(trj.shape[0] - 2)
            ],
            dtype=np.int64,
        )

    def __getitem__(self, idx: int) -> dict:
        ti, t = self.index[idx]
        trj = self.trajs[ti]
        return {"m0": trj[t], "m1": trj[t + 1], "m2": trj[t + 2], "H": self.fields[ti]}

    def __len__(self) -> int:
        return len(self.index)


def dataloader_factory(source, batch_size, num_threads: int = 4, prefetch: int = 4):
    """Factory method for dataloaders for each batch"""

    def closure(seed: int) -> grain.IterDataset:
        return (
            grain.MapDataset.source(source)
            .shuffle(seed=seed)
            .to_iter_dataset(
                grain.ReadOptions(
                    num_threads=num_threads,
                    prefetch_buffer_size=prefetch,
                )
            )
            .batch(batch_size=batch_size, drop_remainder=True)
        )

    return closure
