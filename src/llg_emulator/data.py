from functools import lru_cache

import grain
import msgpack
import numpy as np
from einops import rearrange
from tqdm import tqdm

from llg_emulator.train_config import TrainConfig


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
        self._num_sources = len(source)
        self._decode = decode_fn  # bytes -> np.ndarray (t, c, h, w)
        self._delta = delta
        self.records = self.decode_records()

        self._index = []
        for rec in range(len(self._src)):
            for f in range(T - delta):
                self._index.append((rec, f))

    def decode_records(self):
        records = []
        for src in tqdm(self._src, total=self._num_sources):
            records.append(self._decode(src))
        return records

    def __len__(self):
        return len(self._index)

    def _decode(self, rec):
        return self._decode(self._src[rec])

    def index_dict(self, traj, f):
        return {
            "m0": traj["m"][f],
            "m1": traj["m"][f + self._delta],
            "H_ext": traj["H_ext"],
        }

    def __getitem__(self, idx: int):
        rec, f = self._index[idx]
        traj = self.records[rec]  # (T, C, H, W)
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
