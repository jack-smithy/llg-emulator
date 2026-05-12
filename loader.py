import json
import os

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset, default_collate
import jax


def jax_collate(batch):
    """Collate function to convert a batch of PyTorch data into NumPy arrays."""
    return jtu.tree_map(jnp.asarray, default_collate(batch))


def to_device(batch, device=None):
    """Move a pytree of JAX arrays (or array-likes) to the given JAX device.

    If `device` is None the first device returned by `jax.devices()` is used.
    Call this on batches produced by the `JaxLoader` when training on a single
    device to avoid implicit host-to-device transfers inside the training loop.
    """
    if device is None:
        device = jax.devices()[0]
    return jtu.tree_map(lambda x: jax.device_put(x, device), batch)


class JaxLoader(DataLoader):
    """Custom DataLoader to return Jax arrays from a PyTorch Dataset."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        pin_memory=False,
        drop_last=True,
        generator=None,
        timeout=0,
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


def load_metadata(path):
    with open(f"{path}/params.json") as f:
        p = json.load(f)
    return p


class LLGDataset(Dataset):
    def __init__(self, root_dir, warmup_steps=20):
        # samples holds (path, t) tuples; metadata and memmaps are cached
        self.samples = []  # list of (path, t) pairs
        self.metadata = {}  # path -> params dict
        self.mmaps = {}  # path -> memory-mapped np.ndarray

        for d in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, d)
            p = load_metadata(path=path)
            self.metadata[path] = p
            # keep a persistent mmap object per path to avoid re-opening files
            m = np.load(f"{path}/m.npy", mmap_mode="r")
            self.mmaps[path] = m
            for t in range(warmup_steps, m.shape[0] - 1):
                self.samples.append((path, t))  # pointers to each trajectory pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, t = self.samples[idx]
        p = self.metadata[path]
        m = self.mmaps[path]

        # load only the two timesteps we need
        m_pair = m[t : t + 2, :, :, 0, :]

        # convert to tensor with copy to ensure writable to keep torch happy
        m_pair_tensor = torch.from_numpy(m_pair.copy()).float()

        # rearrange to channel-first for equinox
        m_pair_channel_first = rearrange(m_pair_tensor, "p h w c -> p c h w")
        m0 = m_pair_channel_first[0]  # input magnetization
        target = m_pair_channel_first[1]  # target

        # nondimensionalize external field with Ms
        H_ext = torch.tensor(p["H_ext"]) / p["material"]["Ms"]

        # reshape for broadcasting
        H_ext = repeat(H_ext, "c -> c h w", h=m0.shape[-2], w=m0.shape[-1])
        H_ext = H_ext.float()

        # concat magnetization and field into feature
        feature = torch.concatenate((m0, H_ext), dim=0)
        return feature, target


if __name__ == "__main__":
    dataset = LLGDataset("../micromagnetic-data/data/v2/low_res/train")

    x, y = dataset[100]

    print(x.shape, y.shape)

    loader = JaxLoader(dataset=dataset, batch_size=64, shuffle=True)

    for batch in loader:
        feature, target = batch
        print(feature[:, 3:, 0, 0])
        break
