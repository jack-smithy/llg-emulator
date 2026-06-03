import jax.numpy as jnp
from pathlib import Path
from jaxtyping import Array
from einops import rearrange
from time import perf_counter
from llg_emulator.data import LLGDataset, JaxLoader
import matplotlib.pyplot as plt
import jax
from tqdm import tqdm
import numpy as np
from pdequinox import dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_all_parallel(path: Path, max_workers: int | None = None):
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    results = [None] * len(dirs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_one_trajectory, d): i for i, d in enumerate(dirs)
        }
        for fut in tqdm(as_completed(futures), total=len(dirs)):
            results[futures[fut]] = fut.result()
    return np.concatenate(results, axis=0)


def load_one_trajectory(path: Path) -> Array:
    m: np.ndarray = np.load(path / "m.npy")
    m = m.squeeze(-2)
    m = rearrange(m, "t h w c -> t c h w")
    trjs = slice_windows_of_two(m)
    print(trjs.shape)
    return trjs


def load_all_sequentially(path: Path):
    ms = []
    for dir in tqdm(path.iterdir()):
        ms.append(load_one_trajectory(dir))
    return np.stack(ms, axis=0)


def slice_windows_of_two(one_trj):
    num_time_steps = one_trj.shape[0]

    windows = []
    for i in range(num_time_steps - 1):
        windows.append(one_trj[i : i + 2])

    return jnp.stack(windows)


if __name__ == "__main__":
    path = Path("../micromagnetic-data/data/dynamics/small/train")

    start = perf_counter()
    data = load_all_parallel(path)
    # sub_trj = jax.vmap(slice_windows_of_two)(trj)
    # data = jnp.concatenate(sub_trj)

    for batch in dataloader(data=data, batch_size=32, key=jax.random.PRNGKey(0)):
        pass

    end = perf_counter()
    print(end - start)
    print(data.shape)

    start = perf_counter()
    dataset = LLGDataset(root_dir=path, warmup_steps=0)
    loader = JaxLoader(dataset=dataset, batch_size=32)
    for batch in loader:
        pass

    end = perf_counter()
    print(end - start)
    # m_means = jnp.mean(m, axis=(-1, -2))

    # fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    # for mi in m_means:
    #     axs[0].plot(mi[:, 0])
    #     axs[1].plot(mi[:, 1])
    #     axs[2].plot(mi[:, 2])
    #     axs[0].set_ylim((-1, 1))
    # plt.savefig("data.png")
