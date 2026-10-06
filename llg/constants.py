import math
import os
from pathlib import Path

MU_0 = 4e-7 * math.pi  # magnum.np's value, not the 2019 SI one

# permalloy on 5 x 5 x 3 nm cells, one frame every 10 ps
MATERIAL = {"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02}
DX = (5e-9, 5e-9, 3e-9)
DT = 1e-11
N_STEPS = 100  # frames after the initial one in every trajectory (1 ns)

FIELDS = ("mx", "my", "mz")
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")  # order of `constant_scalars`

DATASETS = Path("datasets")
RESULTS = Path(os.environ.get("RESULTS_ROOT", "results-v2"))


def run_dir(dataset: str, run: str) -> Path:
    return RESULTS / dataset / run


def cache_dir(dataset: str) -> Path:
    return RESULTS / dataset / "scaling"


def coarse_dx(k: int) -> tuple[float, float, float]:
    return (DX[0] * k, DX[1] * k, DX[2])
