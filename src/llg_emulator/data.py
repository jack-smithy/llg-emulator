import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset, default_collate
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import grain


def load_metadata(path) -> dict:
    with open(f"{path}/params.json") as f:
        return json.load(f)


def load_one_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trj: np.ndarray = np.load(path / "m.npy", mmap_mode="r")

    trj = trj.squeeze(-2)  # get rid of singleton z axis
    trj = rearrange(trj, "t h w c -> t c h w")  # channel-first for equinox

    num_time_steps = trj.shape[0]  # num time steps in trajectory

    windows = []  # slice into m_{t}, m_{t+1} windows
    for i in range(num_time_steps - 1):
        windows.append(trj[i : i + 2])
    windows = np.stack(windows)

    params = load_metadata(path)
    H_ext = (
        np.asarray(params["H_ext"])
        * np.ones((windows.shape[0], 3))
        / params["material"]["Ms"]
    )

    return windows, H_ext


def load_all_parallel(path: Path, max_workers: int | None = None):
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    results = [None] * len(dirs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_one_trajectory, d): i for i, d in enumerate(dirs)
        }
        for fut in tqdm(as_completed(futures), total=len(dirs)):
            results[futures[fut]] = fut.result()  # type: ignore

    windows, fields = zip(*results)  # unpack results

    windows = np.concatenate(windows, axis=0)  # type: ignore
    fields = np.concatenate(fields, axis=0)  # type: ignore

    return windows, fields


class LLGSource(grain.sources.RandomAccessDataSource):
    def __init__(self, path: Path, max_workers=16):
        self.windows, self.fields = load_all_parallel(
            path=path,
            max_workers=max_workers,
        )

    def __getitem__(self, idx: int):
        return {"window": self.windows[idx], "field": self.fields[idx]}

    def __len__(self) -> int:
        return self.windows.shape[0]


def make_loader(source, batch_size, seed=0, workers=8):
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=seed,
        num_epochs=None,
        shard_options=grain.sharding.NoSharding(),
    )  # use ShardByJaxProcess() for multi-host
    ops = [grain.transforms.Batch(batch_size, drop_remainder=True)]
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=workers,
        worker_buffer_size=2,
    )  # prefetch on the CPU side


def jax_collate(batch):
    """Collate a batch of PyTorch data into JAX arrays."""
    return jtu.tree_map(jnp.asarray, default_collate(batch))


def to_device(batch, device=None):
    """Move a pytree of array-likes to a JAX device (default: first device)."""
    if device is None:
        device = jax.devices()[0]
    return jtu.tree_map(lambda x: jax.device_put(x, device), batch)


class JaxLoader(DataLoader):
    """DataLoader that yields JAX arrays from a PyTorch Dataset."""

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler=None,
        batch_sampler=None,
        pin_memory: bool = False,
        drop_last: bool = True,
        generator=None,
        timeout: int = 0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=jax_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            generator=generator,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def _nondim_field(p: dict) -> np.ndarray:
    """Applied field nondimensionalized by saturation magnetization Ms.

    Shared by LLGDataset (torch path) and load_trajectory (jax path).
    """
    return np.asarray(p["H_ext"], dtype=np.float32) / p["material"]["Ms"]


# Time-axis convention: frame 0 is an identical duplicate of frame 1 (the
# initial condition) and is dropped everywhere. Index 0 of every trajectory we
# expose is raw frame 1.
class LLGDataset(Dataset):
    def __init__(self, root_dir, warmup_steps: int = 20):
        self.samples = []  # list of (path, t) pairs
        self.metadata = {}  # path -> params dict
        self.mmaps = {}  # path -> memory-mapped np.ndarray

        for d in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, d)
            p = load_metadata(path=path)
            self.metadata[path] = p
            # persistent zero-copy mmap with frame 0 dropped
            m = np.load(f"{path}/m.npy", mmap_mode="r")[1:]
            self.mmaps[path] = m
            for t in range(warmup_steps, m.shape[0] - 1):
                self.samples.append((path, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, t = self.samples[idx]
        p = self.metadata[path]
        m = self.mmaps[path]

        # only the two timesteps we need; drop the z singleton axis
        m_pair = m[t : t + 2, :, :, 0, :]
        m_pair_tensor = torch.from_numpy(m_pair.copy()).float()
        m_pair_channel_first = rearrange(m_pair_tensor, "p h w c -> p c h w")
        m0 = m_pair_channel_first[0]  # input magnetization
        target = m_pair_channel_first[1]  # target

        H_ext = torch.from_numpy(_nondim_field(p))
        H_ext = repeat(H_ext, "c -> c h w", h=m0.shape[-2], w=m0.shape[-1]).float()

        feature = torch.concatenate((m0, H_ext), dim=0)
        return feature, target


def load_trajectory(path: Path):
    """Load a full trajectory (jax) + its nondimensionalized applied field.

    Returns (m, H_ext) with m shape (T, 3, H, W); same time-axis convention as
    LLGDataset (index 0 = raw frame 1).
    """
    m = jnp.load(path / "m.npy", mmap_mode="r")
    m = m[1:, :, :, 0, :]
    m = rearrange(m, "t h w c -> t c h w")

    with open(path / "params.json", "r") as f:
        p = json.load(f)

    return m, jnp.asarray(_nondim_field(p))


if __name__ == "__main__":
    path = Path("../micromagnetic-data/data/dynamics/small/train")
    source = LLGSource(path=path, max_workers=4)

    loader = make_loader(source=source, batch_size=512)

    for batch in loader:
        print(batch["window"].shape)
