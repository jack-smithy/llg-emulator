"""Where runs and evaluation caches live: `results-v2/` unless the environment sets
`RESULTS_ROOT` (e.g. `RESULTS_ROOT=results-closure` for the closure sweeps), so one
checkout can keep separate result trees without touching the CLIs. Every script joins
`RESULTS / <dataset> / ...` from here; the dataset's `scaling/` cache under a root may
be a symlink to a shared one."""

import os
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_ROOT", "results-v2"))
