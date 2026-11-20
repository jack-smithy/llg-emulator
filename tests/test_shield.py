"""Checks for the shield/ modules. Run inside a job, e.g.
srun -p devcpu -c 4 -t 15 uv run pytest -q
"""

import h5py
import magpylib
import numpy as np
import pytest
import torch
from scipy import constants
from the_well.data import WellDataset

from shield import coarsen, geometry, protocols, strayfield, well_io


def test_block_average_conserves_moment_and_composes():
    m = torch.randn(2, 8, 8, 4, 3, dtype=torch.float64)  # (frames, nx, ny, nz, 3)
    coarse = coarsen.block_average(m, 4)
    assert coarse.shape == (2, 2, 2, 1, 3)
    torch.testing.assert_close(coarse.sum((1, 2, 3)) * 4**3, m.sum((1, 2, 3)))
    torch.testing.assert_close(coarsen.block_average(coarsen.block_average(m, 2), 2), coarse)

    uniform = torch.ones(4, 4, 4, 3)
    assert torch.equal(coarsen.block_average(uniform, 2), torch.ones(2, 2, 2, 3))

    with pytest.raises(ValueError):
        coarsen.block_average(m, 3)


def test_shapes():
    prism = geometry.volume_fraction("prism", (100e-9, 50e-9, 20e-9), 5e-9)
    assert prism.shape == (20, 10, 4) and prism.min() == 1
    disk = geometry.volume_fraction("ellipse", (320e-9, 320e-9, 5e-9), 5e-9)[:, :, 0]  # 64 x 64 cells
    np.testing.assert_allclose(disk.sum(), np.pi * 32**2, rtol=1e-2)
    np.testing.assert_array_equal(disk, disk[::-1, ::-1])
    with pytest.raises(ValueError):
        geometry.n_cells((101e-9, 50e-9, 20e-9), 5e-9)


def test_sweep_and_initial_states():
    b = protocols.sweep(protocols.major_loop(0.2), direction=(0, 2, 0))
    amplitude = b[:, 1]
    assert amplitude[0] == 0 and amplitude[-1] == pytest.approx(0.2)
    assert amplitude.min() == pytest.approx(-0.2)
    assert np.abs(np.diff(amplitude)).max() <= protocols.STEP + 1e-12
    assert np.all(b[:, [0, 2]] == 0)

    rng = np.random.default_rng(0)
    for kind in ("random", "vortex", "+x", "-y"):
        m = protocols.initial_state(kind, (6, 4, 2), rng)
        np.testing.assert_allclose(np.linalg.norm(m, axis=-1), 1)


def test_well_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n_steps, n = 5, (6, 4, 2)
    dx = 20e-9
    for split in ("train", "valid"):
        (tmp_path / "data" / split).mkdir(parents=True)
        for i in range(2):
            m = rng.uniform(-1, 1, size=(1, n_steps, *n, 3))
            well_io.write_well_file(
                tmp_path / "data" / split / f"trajectory_{i}.hdf5",
                dx=dx,
                m=m,
                mask=np.ones((1, *n)),
                e_ex=rng.uniform(0, 1e3, size=(1, n_steps, *n)),
                step_scalars={"Bx": np.zeros((1, n_steps)), "By": np.linspace(0, 0.1, n_steps)[None]},
                trajectory_scalars={"Lx": np.array([n[0] * dx])},
                constants={"Ms": geometry.MS, "A": geometry.A_EX},
            )

    dataset = WellDataset(path=str(tmp_path), well_split_name="train", n_steps_input=1, n_steps_output=1)
    assert len(dataset) == 2 * (n_steps - 1)
    sample = dataset[1]
    assert sample["input_fields"].shape == (1, *n, 4)  # e_ex, m_x, m_y, m_z
    assert sample["constant_fields"].shape == (*n, 1)  # mask
    assert sample["input_scalars"].shape == (1, 2)  # Bx, By
    # .cpu(): importing magnum.np (via strayfield) makes the GPU torch's default device
    assert sample["output_scalars"][0, 1].item() == pytest.approx(0.05)  # By at step 2
    assert torch.all(sample["boundary_conditions"] == 1)  # open on every side
    with h5py.File(tmp_path / "data/train/trajectory_0.hdf5") as f:
        np.testing.assert_allclose(sample["output_fields"][0, ..., 1:].cpu(), f["t1_fields/m"][0, 2], rtol=1e-6)


def cell_average(cuboid, centres, dx, order=6):
    """magpylib's H averaged over cubic cells (k, 3) by Gauss-Legendre quadrature; returns (k, 3)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    offsets = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3) * dx / 2
    cell_weights = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1) / 8
    points = centres[:, None, :] + offsets[None]  # (k, order^3, 3)
    H = cuboid.getH(points.reshape(-1, 3)).reshape(len(centres), -1, 3)
    return (H * cell_weights[None, :, None]).sum(axis=1)


def test_stray_field_matches_analytic_cuboid():
    """Uniformly magnetised prism: magnum.np demag on an air-padded grid vs magpylib's closed form."""
    n, dx = (20, 10, 20), 10e-9
    size = [k * dx for k in n]
    M = np.zeros((*n, 3))
    M[..., 1] = geometry.MS
    pad = strayfield.probe_padding(n)
    H = strayfield.stray_field_operator(n, dx, pad)(M)
    cuboid = magpylib.magnet.Cuboid(dimension=size, polarization=(0, constants.mu_0 * geometry.MS, 0))

    # whole air cells above and below the magnet: magnum.np's value is the exact cell average
    top, bottom = pad[2] + n[2], pad[2] - 1  # first air layer above / below
    cells = np.array([(10, pad[1] + 5, top + 1), (3, pad[1] + 2, top + 4), (10, pad[1] + 5, bottom - 2)])
    centres = (cells + 0.5 - np.array(H.shape[:3]) / 2) * dx
    expected = cell_average(cuboid, centres, dx)
    actual = H[tuple(torch.as_tensor(cells.T))].cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5 * np.abs(expected).max())

    # probes interpolate cell averages; here the h = 0.1 probes are only two cells from the surface
    points = strayfield.probe_points(size)
    H_probes = strayfield.sample(H, dx, points)
    H_exact = cuboid.getH(points)
    np.testing.assert_allclose(H_probes, H_exact, rtol=0.02, atol=1e-3 * np.abs(H_exact).max())


def test_stray_field_independent_of_padding():
    n, dx = (8, 6, 4), 10e-9
    rng = np.random.default_rng(0)
    M = rng.normal(size=(*n, 3)) * geometry.MS
    points = strayfield.probe_points([k * dx for k in n])
    pad = strayfield.probe_padding(n)
    more_pad = (pad[0] + 3, pad[1] + 2, pad[2] + 4)
    H_small = strayfield.sample(strayfield.stray_field_operator(n, dx, pad)(M), dx, points)
    H_large = strayfield.sample(strayfield.stray_field_operator(n, dx, more_pad)(M), dx, points)
    np.testing.assert_allclose(H_small, H_large, rtol=1e-9, atol=1e-9 * np.abs(H_small).max())


def test_demag_is_differentiable():
    n, dx = (4, 3, 2), 10e-9
    field = strayfield.stray_field_operator(n, dx, pad=(1, 0, 1))
    M = torch.randn(*n, 3, dtype=torch.float64, device=torch.get_default_device(), requires_grad=True)
    assert torch.autograd.gradcheck(field, (M,))
