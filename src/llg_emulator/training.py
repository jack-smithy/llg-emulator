"""Training core: the loss, the jitted update step, and the LR schedule.

The objective is a plain **one-step MSE** on `(m_t, m_{t+stride})` pairs; the
step size the model is conditioned on rides along in the batch as `s0`. The
demag tensor is held out of the gradient by `trainable_filter`, which is also
what `io.save_model` filters on so the frozen tensor never reaches a checkpoint.
"""

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import PyTree

from llg_emulator.model import LLGEmulator


def trainable_filter(model: LLGEmulator):
    spec = jtu.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: m.demag.N, spec, replace=False)


def count_parameters(model: LLGEmulator) -> int:
    return sum(
        p.size for p in jtu.tree_leaves(eqx.filter(model, trainable_filter(model)))
    )


@eqx.filter_jit
def loss_fn(model, m0, m1, m2, H, s0, pushforward: bool = False):
    """One-step MSE, or the pushforward variant.

    Plain: match `m1` from the true `m0`. The model then only ever sees exact
    inputs, while a rollout feeds it its own slightly-wrong states -- on SP4 that
    distribution shift turns a 0.002 one-step error into a 0.056 rollout error.

    Pushforward (Brandstetter et al.): take one step, **detach it**, and match
    `m2` from there. The input is now drawn from the model's own error
    distribution, but the gradient still crosses a single step, so it costs one
    extra forward pass rather than a k-step backward.
    """
    cond = jnp.concat((H, jnp.expand_dims(s0, -1)), axis=-1)
    if pushforward:
        m0 = jax.lax.stop_gradient(jax.vmap(model)(m0, cond))
        m1 = m2
    m1_pred = jax.vmap(model)(m0, cond)
    return jnp.mean(jnp.square(m1_pred - m1))


@eqx.filter_jit(donate="all")
def update_fn(model: LLGEmulator, batch: PyTree, optimizer, opt_state, pushforward=False):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), pushforward=pushforward, **batch)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state, params=diff)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader, optimizer, opt_state, device, pushforward=False):
    losses = []
    for batch in loader:
        batch = jax.device_put(batch, device)
        model, opt_state, loss = update_fn(model, batch, optimizer, opt_state, pushforward)
        losses.append(loss)
    return model, opt_state, jnp.stack(losses).mean().item()


def val_epoch(model, loader: grain.IterDataset) -> float:
    """One-step MSE on the val split -- always plain, so the number stays
    comparable across runs whatever the training loss is."""
    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in loader:
        losses.append(loss_fn(inference_model, **batch))
    return jnp.stack(losses).mean().item()


def make_schedule(lr, epochs, steps_per_epoch, warmup_frac=0.05):
    """Warmup + cosine decay over the whole run."""
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * warmup_frac))
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=lr * 0.02,
    )


def _demo():
    """Self-check: the losses wire up and the pushforward detach is real."""
    import jax.random as jr

    from llg_emulator.model import LLGEmulator, ModelConfig
    from llg_emulator.physics import demag_for

    cfg = ModelConfig(hidden_channels=8, num_blocks=2, mesh_n=(15, 15))
    demag = demag_for(cfg.mesh_n, cfg.mesh_dx, 1.0, cfg.demag_p)
    model = LLGEmulator(config=cfg, demag=demag, key=jr.PRNGKey(0))

    def unit(key):
        v = jr.normal(key, (2, 3, 16, 16))
        return v / jnp.linalg.norm(v, axis=1, keepdims=True)

    m0, m1, m2 = (unit(k) for k in jr.split(jr.PRNGKey(1), 3))
    H, s0 = jnp.zeros((2, 3)), jnp.zeros((2,))

    # the model is the identity at init, so each loss reduces to a plain MSE and
    # the pushforward variant is visibly aimed two frames ahead, not one
    assert jnp.allclose(loss_fn(model, m0, m1, m2, H, s0), jnp.mean((m0 - m1) ** 2), atol=1e-5)
    assert jnp.allclose(
        loss_fn(model, m0, m1, m2, H, s0, pushforward=True),
        jnp.mean((m0 - m2) ** 2),
        atol=1e-5,
    )

    # Without stop_gradient this would be a 2-step rollout loss, which trains a
    # different thing at k times the backward cost. Compare the two gradients:
    # if the detach ever gets dropped, they collapse onto each other.
    diff, static = eqx.partition(model, trainable_filter(model))
    cond = jnp.concat((H, jnp.expand_dims(s0, -1)), axis=-1)

    def two_step(d):
        m = eqx.combine(d, static)
        return jnp.mean(jnp.square(jax.vmap(m)(jax.vmap(m)(m0, cond), cond) - m2))

    g_sg = eqx.filter_grad(
        lambda d: loss_fn(eqx.combine(d, static), m0, m1, m2, H, s0, pushforward=True)
    )(diff)
    g_full = eqx.filter_grad(two_step)(diff)

    flat = lambda g: jnp.concatenate([x.ravel() for x in jtu.tree_leaves(g)])
    a, b = flat(g_sg), flat(g_full)
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    assert jnp.all(jnp.isfinite(a)) and jnp.linalg.norm(a) > 0, "no pushforward gradient"
    assert rel > 1e-3, f"detach is not doing anything: grads differ by {rel:.2e}"
    print(f"training self-check ok: pushforward detach cuts {rel:.1%} of the 2-step gradient")


if __name__ == "__main__":
    _demo()
