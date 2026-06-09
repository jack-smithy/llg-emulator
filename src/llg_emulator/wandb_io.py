"""Cloud lookup/download helpers shared by evaluate.py and animate.py.

Training logs each run's weights as a single wandb model artifact named
``model-<run_id>`` (containing every ``weights_epoch_*.eqx`` plus the final
``weights.eqx``). These helpers resolve a run by its human-friendly display
name and fetch that artifact from the cloud.
"""

import wandb
import tempfile
from pathlib import Path
import equinox as eqx
from dataclasses import asdict


def model_artifact_name(run_id: str) -> str:
    """Name of the per-run model artifact (one artifact, all checkpoints)."""
    return f"model-{run_id}"


def find_run(name: str, project: str, entity: str | None = None):
    """Resolve a wandb run by its display name (e.g. 'lively-firefly-3').

    Returns the public API run (carrying ``.id`` and ``.config``). Raises if no
    run or more than one run matches the name.
    """
    api = wandb.Api()
    entity = entity or api.default_entity
    path = f"{entity}/{project}"
    runs = list(api.runs(path, filters={"display_name": name}))
    if not runs:
        raise ValueError(f"no run named {name!r} in {path}")
    if len(runs) > 1:
        ids = ", ".join(r.id for r in runs)
        raise ValueError(
            f"{len(runs)} runs named {name!r} in {path} (ids: {ids}); "
            "the name is ambiguous"
        )
    return runs[0]


def download_model_dir(run) -> str:
    """Download a run's model artifact to a local cache dir and return its path.

    ``run`` may be a public API run (from find_run) or the active
    ``wandb.run``; either way the latest ``model-<id>`` artifact is fetched.
    """
    artifact_ref = "weights:v0"
    if wandb.run is not None and run.id == wandb.run.id:
        artifact = wandb.run.use_artifact(artifact_ref)
    else:
        api = wandb.Api()
        artifact = api.artifact(f"{run.entity}/{run.project}/{artifact_ref}")
    return artifact.download()


def save_weights(model, step=None, aliases=None):
    assert wandb.run is not None
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "weights.eqx"
        eqx.tree_serialise_leaves(path, model)
        artifact = wandb.Artifact(
            "weights",
            type="model",
            metadata={
                "run_id": wandb.run.id,
                "run_name": wandb.run.name,
                "step": step,
            },
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact, aliases=aliases)
        artifact.wait()
