import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import grain
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array
from tqdm import tqdm


def load_metadata(path) -> dict:
    with open(f"{path}/params.json") as f:
        return json.load(f)


def load_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def load_trajectory(path: Path) -> tuple[Array, Array]:
    trj = jnp.load(path / "m.npy", mmap_mode="r")
    trj = trj.squeeze(-2)
    trj = rearrange(trj, "t h w c -> t c h w")

    params = load_metadata(path)
    H_ext = jnp.asarray(params["H_ext"]) / params["material"]["Ms"]
    return trj, H_ext


def load_windows_parallel(path: Path, max_workers: int | None = None):
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_windows, d) for d in dirs]
        results = [f.result() for f in tqdm(as_completed(futures), total=len(dirs))]

    windows, fields = zip(*results)
    return np.concatenate(windows), np.concatenate(fields)


class LLGStepperSource(grain.sources.RandomAccessDataSource):
    def __init__(self, path: Path, max_workers=16):
        self.windows, self.fields = load_windows_parallel(
            path=path,
            max_workers=max_workers,
        )

    def __getitem__(self, idx: int):
        m0 = self.windows[idx][0]
        m1 = self.windows[idx][1]
        return {"m0": m0, "m1": m1, "H": self.fields[idx]}

    def __len__(self) -> int:
        return self.windows.shape[0]


def dataloader_factory(source, batch_size):
    def closure(seed: int) -> grain.IterDataset:
        return (
            grain.MapDataset.source(source=source)
            .shuffle(seed=seed)
            .to_iter_dataset()
            .batch(batch_size=batch_size, drop_remainder=True)
        )

    return closure
