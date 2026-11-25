"""Write quasi-static trajectories in the Well's HDF5 format (from shield/well_io.py, commit 79d92da).

The Well loads one grid shape per split, so all files of a split share (nx, ny, nz). Layout of
one file (n = trajectories, T = field steps):

    t1_fields/m     (n, T, nx, ny, nz, 3)  magnetisation in units of Ms (|m| = 1)
    scalars         constants (), per trajectory (n,), per step (n, T): e.g. Bx, By, Bz
    dimensions      x, y, z cell centres (m, centred on 0); time = field-step index

Boundaries are open: a magnet in vacuum.
"""

from pathlib import Path

import h5py
import numpy as np
import yaml

DATASET_NAME = "shield"


def _add(group, name, data, sample_varying, time_varying, **dataset_options):
    dataset = group.create_dataset(name, data=data, **dataset_options)
    dataset.attrs["sample_varying"] = sample_varying
    dataset.attrs["time_varying"] = time_varying
    return dataset


def write_well_file(path, dx, m, step_scalars, trajectory_scalars, constants):
    """Write one Well file.

    dx: cell size (m). m: (n, T, nx, ny, nz, 3), an array or an HDF5 dataset.
    step_scalars: name -> (n, T); trajectory_scalars: name -> (n,); constants: name -> float.
    """
    n_trajectories, n_steps, nx, ny, nz, _ = m.shape
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = DATASET_NAME
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 3
        f.attrs["n_trajectories"] = n_trajectories
        f.attrs["simulation_parameters"] = list(constants) + list(trajectory_scalars)

        dimensions = f.create_group("dimensions")
        dimensions.attrs["spatial_dims"] = ["x", "y", "z"]
        _add(dimensions, "time", np.arange(n_steps, dtype=np.float32), False, True)
        for axis, n in zip("xyz", (nx, ny, nz)):
            centres = (np.arange(n) + 0.5) * dx - n * dx / 2
            _add(dimensions, axis, centres.astype(np.float32), False, False)

        boundaries = f.create_group("boundary_conditions")
        for axis, n in zip("xyz", (nx, ny, nz)):
            group = boundaries.create_group(f"{axis}_open")
            group.attrs["associated_dims"] = [axis]
            group.attrs["associated_fields"] = ["m"]
            group.attrs["bc_type"] = "OPEN"
            group.attrs["sample_varying"] = False
            group.attrs["time_varying"] = False
            edges = np.zeros(n, dtype=bool)
            edges[[0, -1]] = True
            group.create_dataset("mask", data=edges)

        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = list(constants) + list(trajectory_scalars) + list(step_scalars)
        for name, value in constants.items():
            _add(scalars, name, np.float32(value), False, False)
        for name, value in trajectory_scalars.items():
            _add(scalars, name, np.asarray(value, dtype=np.float32), True, False)
        for name, value in step_scalars.items():
            _add(scalars, name, np.asarray(value, dtype=np.float32), True, True)

        f.create_group("t0_fields").attrs["field_names"] = []

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = ["m"]
        # one frame per chunk, so the loader reads only the steps it needs; written frame by frame,
        # so m may also be an HDF5 dataset larger than memory
        m_dataset = t1.create_dataset(
            "m", shape=m.shape, dtype=np.float32, chunks=(1, 1, nx, ny, nz, 3), compression="lzf"
        )
        for t in range(n_steps):
            m_dataset[:, t] = m[:, t]
        m_dataset.attrs["sample_varying"] = True
        m_dataset.attrs["time_varying"] = True
        m_dataset.attrs["dim_varying"] = [True, True, True]

        f.create_group("t2_fields").attrs["field_names"] = []


def write_stats(dataset_dir):
    """Write <dataset_dir>/stats.yaml from the train split, per component of m.

    Keys as the Well's normalisations expect them: mean/std/rms of m and of its step-to-step
    change (*_delta), each a list [x, y, z]. One file in memory at a time.
    """
    dataset_dir = Path(dataset_dir)
    sums = {"m": np.zeros(3), "m2": np.zeros(3), "d": np.zeros(3), "d2": np.zeros(3)}
    n_values, n_deltas = 0, 0
    for path in sorted((dataset_dir / "data" / "train").glob("*.hdf5")):
        with h5py.File(path) as f:
            m = f["t1_fields/m"][:].astype(np.float64)  # (n, T, nx, ny, nz, 3)
        delta = np.diff(m, axis=1)
        sums["m"] += m.reshape(-1, 3).sum(axis=0)
        sums["m2"] += (m**2).reshape(-1, 3).sum(axis=0)
        sums["d"] += delta.reshape(-1, 3).sum(axis=0)
        sums["d2"] += (delta**2).reshape(-1, 3).sum(axis=0)
        n_values += m.size // 3
        n_deltas += delta.size // 3

    mean, mean_square = sums["m"] / n_values, sums["m2"] / n_values
    mean_delta, mean_square_delta = sums["d"] / n_deltas, sums["d2"] / n_deltas
    stats = {
        "mean": {"m": mean.tolist()},
        "std": {"m": np.sqrt(np.maximum(mean_square - mean**2, 0)).tolist()},
        "rms": {"m": np.sqrt(mean_square).tolist()},
        "mean_delta": {"m": mean_delta.tolist()},
        "std_delta": {"m": np.sqrt(np.maximum(mean_square_delta - mean_delta**2, 0)).tolist()},
        "rms_delta": {"m": np.sqrt(mean_square_delta).tolist()},
    }
    (dataset_dir / "stats.yaml").write_text(yaml.safe_dump(stats, sort_keys=True))
