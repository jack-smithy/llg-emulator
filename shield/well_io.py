"""Write quasi-static shield trajectories in the Well's HDF5 format.

The Well loads one grid shape per split, so all files of a split share (nx, ny, nz). Layout of
one file (n = trajectories, T = field steps):

    t1_fields/m     (n, T, nx, ny, nz, 3)  moment density in units of Ms (|m| <= 1, 0 in air)
    t0_fields/mask  (n, nx, ny, nz)        magnetic volume fraction, constant in time
    t0_fields/e_ex  (n, T, nx, ny, nz)     exchange energy density (J/m^3) of the fine state
    scalars         constants (), per trajectory (n,), per step (n, T): Bx, By, Bz, energies, probes
    dimensions      x, y, z cell centres (m, centred on 0); time = field-step index

Boundaries are open: a magnet in vacuum.
"""

from pathlib import Path

import h5py
import numpy as np

from shield.geometry import cell_centres

DATASET_NAME = "nife_shield_quasistatic"


def _add(
    group: h5py.Group,
    name: str,
    data: np.ndarray | np.generic,
    sample_varying: bool,
    time_varying: bool,
    **dataset_options,
) -> h5py.Dataset:
    dataset = group.create_dataset(name, data=data, **dataset_options)
    dataset.attrs["sample_varying"] = sample_varying
    dataset.attrs["time_varying"] = time_varying
    return dataset


def _add_field(group: h5py.Group, name: str, data: np.ndarray, time_varying: bool) -> None:
    """A per-trajectory field, chunked one frame per chunk."""
    leading = (1, 1) if time_varying else (1,)
    frame_shape = data.shape[len(leading):]
    dataset = _add(
        group, name, np.asarray(data, dtype=np.float32), True, time_varying,
        chunks=leading + frame_shape, compression="lzf",
    )
    dataset.attrs["dim_varying"] = [True, True, True]


def write_well_file(
    path: str | Path,
    dx: float,
    m: np.ndarray,
    mask: np.ndarray,
    e_ex: np.ndarray,
    step_scalars: dict[str, np.ndarray],
    trajectory_scalars: dict[str, np.ndarray],
    constants: dict[str, float],
) -> None:
    """Write one Well file.

    dx: stored cell size (m). m (n, T, nx, ny, nz, 3), mask (n, nx, ny, nz), e_ex (n, T, nx, ny, nz).
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
            _add(dimensions, axis, cell_centres(n, dx).astype(np.float32), False, False)

        boundaries = f.create_group("boundary_conditions")
        for axis, n in zip("xyz", (nx, ny, nz)):
            group = boundaries.create_group(f"{axis}_open")
            group.attrs["associated_dims"] = [axis]
            group.attrs["associated_fields"] = ["m", "e_ex"]
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

        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = ["mask", "e_ex"]
        _add_field(t0, "mask", mask, time_varying=False)
        _add_field(t0, "e_ex", e_ex, time_varying=True)

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = ["m"]
        _add_field(t1, "m", m, time_varying=True)

        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []
