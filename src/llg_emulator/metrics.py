import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from llg_emulator.model import LLGEmulator
from llg_emulator.rollout import rollout_trajectory, rollout_trajectories
from llg_emulator.data import LLGStepperSource, load_metadata, load_trajectory
from llg_emulator.physics import DemagField
import numpy as np


def nRMSE(pred, ref) -> Array:
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def correlation(pred, ref):
    assert pred.shape == ref.shape
    assert len(pred.shape) >= 4
    # both shape (..., t, c, nx, ny)
    pointwise_corr = jnp.sum(pred * ref, axis=-3)  # (..., t, nx, ny)
    framewise_corr = jnp.sqrt(jnp.mean(pointwise_corr**2, axis=(-1, -2)))  # (..., t,)
    if len(framewise_corr.shape) == 2:  # if batch axis, reduce over batches
        return framewise_corr.mean(axis=0), framewise_corr.std(axis=0)
    return framewise_corr, jnp.zeros_like(framewise_corr)


def MSE(pred: Array, ref: Array) -> Array:
    return jnp.mean(jnp.square(pred - ref))


def correlation_epoch(model, source):

    def loader(source, batch_size):
        num_trjs = len(source.fields)
        num_batches = num_trjs // batch_size

        for i in range(num_batches):
            yield (
                np.stack(source.trajs[i : i + batch_size]),
                np.stack(source.fields[i : i + batch_size]),
            )

    m_refs, m_preds = [], []
    for m_ref, field in loader(source, 4):
        m_pred = rollout_trajectories(model, m_ref, field)
        m_preds.append(m_pred)
        m_refs.append(m_ref)

    m_ref = jnp.concat(m_refs, axis=0)
    m_pred = jnp.concat(m_preds, axis=0)

    mean, std = correlation(m_pred, m_ref)
    return mean, std


def bulk_rollout(model, m0, H, n_steps, s0=0.0):
    """Per-step spatial mean of a rollout, without materialising the frames.

    Equivalent to the mean of `rollout_trajectory`, but the scan carries only
    the current frame: on the 2048^2 domain the full history is ~5 GB of device
    memory to produce a (t, 3) curve.
    """
    cond = jnp.concatenate((H, jnp.array([s0])))

    def scan_fn(m, _):
        m_next = model(m, cond)
        return m_next, jnp.mean(m_next, axis=(1, 2))

    _, means = jax.lax.scan(scan_fn, m0, None, length=n_steps)
    return jnp.concatenate([jnp.mean(m0, axis=(1, 2))[None], means], axis=0)


def bulk_magnetization(model, path):
    """Reference vs. rolled-out bulk magnetization for one trajectory.

    Rebuilds the demag tensor when the trajectory's mesh differs from the one
    the model was built with (the domain-size-invariance check): the ResNet
    backbone is resolution-agnostic, the demag kernel is mesh-shaped.
    """
    m_true, H_ext = load_trajectory(path)
    meta = load_metadata(path)

    if tuple(meta["n"]) != model.demag.n:
        # ponytail: p is not stored on DemagField; 20 is the only value used.
        demag = DemagField(meta["n"], meta["dx"], Ms=model.demag.Ms, p=20)
        model = eqx.tree_at(lambda m: m.demag, model, demag)

    bulk_pred = bulk_rollout(
        eqx.nn.inference_mode(model),
        jnp.asarray(m_true[0]),
        jnp.asarray(H_ext),
        n_steps=m_true.shape[0] - 1,
    )
    return m_true.mean(axis=(2, 3)), bulk_pred


def sp4_rollout_rmse(model: LLGEmulator, source: LLGStepperSource) -> float:
    """Bulk-magnetization RMSE of a 100-step SP4 rollout vs. the reference.

    The model-selection metric: SP4 is the deliverable and its applied field is
    out-of-distribution, so this is what best-checkpointing tracks.
    """
    model = eqx.nn.inference_mode(model)

    m_true, H = source.trajs[0], source.fields[0]
    m_true = jnp.asarray(m_true)
    m_pred = rollout_trajectory(model, m_true, jnp.asarray(H), include_init=True)
    bulk_true = jnp.mean(m_true, axis=(2, 3))
    bulk_pred = jnp.mean(m_pred, axis=(2, 3))
    return float(jnp.sqrt(jnp.mean((bulk_true - bulk_pred) ** 2)))
