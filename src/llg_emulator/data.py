import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import grain
import msgpack
import numpy as np
from einops import rearrange
from tqdm import tqdm

from llg_emulator.train_config import TrainConfig


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


def decode_record(raw: bytes) -> dict:
    record = msgpack.unpackb(raw, raw=False)
    m = np.frombuffer(record["m"], dtype=np.float32).reshape(record["shape"])
    m = rearrange(m, "... h w 1 c -> ... c h w")
    Ms = record["params"]["Ms"]
    H_ext = np.asarray(record["H_ext"], dtype=np.float32)
    return {"m": m, "Ms": Ms, "H_ext": H_ext}


class LLGStepperSource:
    def __init__(self, source, T=101, decode_fn=decode_record, delta=1):
        self._src = source
        self._decode = decode_fn  # bytes -> np.ndarray (t, c, h, w)
        self._delta = delta

        self._index = []
        for rec in range(len(self._src)):
            for f in range(T - delta):
                self._index.append((rec, f))

    def __len__(self):
        return len(self._index)

    @lru_cache(32)
    def _decode_cached(self, rec):
        return self._decode(self._src[rec])

    def index_dict(self, traj, f):
        return {
            "m0": traj["m"][f],
            "m1": traj["m"][f + self._delta],
            "H_ext": traj["H_ext"],
        }

    def __getitem__(self, idx: int):
        rec, f = self._index[idx]
        traj = self._decode_cached(rec)  # (T, C, H, W)
        return self.index_dict(traj, f)


class TrajectoryDataSource:
    def __init__(self, source, decode_fn=decode_record):
        self._src = source
        self._decode = decode_fn

    def __len__(self):
        return len(self._src)

    @lru_cache(32)
    def _decode_cached(self, idx):
        return self._decode(self._src[idx])

    def __getitem__(self, idx: int):
        return self._decode_cached(idx)


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
