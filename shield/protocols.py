"""Field histories and initial states for quasi-static sweeps.

A trajectory starts from an initial state and steps the applied field through a list of
turning points; after every step the magnetisation is relaxed, so each frame is an
equilibrium state that remembers the field history.
"""

import itertools
from collections.abc import Sequence

import numpy as np

STEP = 5e-3  # T, largest field step


def sweep(
    turning_points: Sequence[float], direction: Sequence[float], step: float = STEP
) -> np.ndarray:
    """Applied field (n_steps, 3) in tesla, stepping linearly through `turning_points`.

    turning_points: field amplitudes in tesla along `direction`, e.g. [0, 0.2, -0.2, 0.2] for a
    major loop. No step is larger than `step`; the first frame is the first turning point.
    """
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    amplitudes = [turning_points[0]]
    for start, end in itertools.pairwise(turning_points):
        n_steps = max(1, int(np.ceil(abs(end - start) / step - 1e-9)))
        amplitudes.extend(np.linspace(start, end, n_steps + 1)[1:])
    return np.outer(amplitudes, unit)


def major_loop(b_max: float) -> list[float]:
    return [0.0, b_max, -b_max, b_max]


def reversal_curve(b_max: float, b_reverse: float) -> list[float]:
    """Saturate, come back to b_reverse, saturate again (a first-order reversal curve)."""
    return [0.0, b_max, b_reverse, b_max]


def operating_point(b_history: float, b_operate: float = 0.05) -> list[float]:
    """Go to b_history first, then to the operating field (the PDF relaxes at 50 mT)."""
    return [0.0, b_history, b_operate]


def ac_demagnetisation(b_max: float, decay: float = 0.8) -> list[float]:
    """Alternating field with decaying amplitude, ending at zero field."""
    points = [0.0]
    amplitude = b_max
    sign = 1
    while amplitude > STEP:
        points.append(sign * amplitude)
        amplitude *= decay
        sign = -sign
    points.append(0.0)
    return points


def initial_state(kind: str, n: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Unit magnetisation (nx, ny, nz, 3) on a grid of n = (nx, ny, nz) cells.

    kind: "random", "vortex" (flux closure m ~ (-y, x, 0) with an out-of-plane core one cell
    wide, the PDF's Landau/vortex start) or a uniform direction "+x", "-x", "+y", "-y".
    """
    nx, ny, nz = n
    uniform = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}

    if kind == "random":
        m = rng.normal(size=(nx, ny, nz, 3))
    elif kind == "vortex":
        x = np.arange(nx) + 0.5 - nx / 2  # cell centres in units of cells
        y = np.arange(ny) + 0.5 - ny / 2
        x, y = np.meshgrid(x, y, indexing="ij")
        in_plane = np.stack([-y, x, np.ones_like(x)], axis=-1)
        m = np.repeat(in_plane[:, :, None, :], nz, axis=2)
    elif kind in uniform:
        m = np.zeros((nx, ny, nz, 3))
        m[...] = uniform[kind]
    else:
        raise ValueError(f"unknown initial state {kind!r}")

    return m / np.linalg.norm(m, axis=-1, keepdims=True)
