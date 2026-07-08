"""Self-contained trainer with weight saving, for benchmarking experiments.

Mirrors the wandb training loop in __init__.py but (a) saves the model via
bench.save_model so it can be reloaded/evaluated and (b) exposes the knobs we
sweep (lr schedule, epochs, arch, and an optional multi-step rollout loss that
directly targets autoregressive stability). No wandb.

    uv run python -m llg_emulator.train_bench --out weights.eqx --tag myrun ...
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from tqdm import tqdm

from llg_emulator.bench import append_benchmark, evaluate, load_model, save_model
from llg_emulator.config import dataset_dir
from llg_emulator.data import LLGStepperSource, dataloader_factory, load_trajectories
from llg_emulator.metrics import MSE
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.symmetry import NUM_ELEMENTS, apply_field, apply_H
from llg_emulator.train_config import TrainConfig
from llg_emulator.training import trainable_filter


def rollout_loss_fn(model, m_seq, H, k: int):
    """k-step unrolled MSE for one trajectory chunk.

    m_seq: (k+1, 3, nx, ny) consecutive ground-truth frames. We feed the model
    its own prediction for k steps and match every intermediate frame, so the
    gradient sees compounding error — the failure mode of one-step training.
    """
    def scan_fn(m, m_target):
        m_next = model(m, H)
        return m_next, MSE(m_next, m_target)

    _, losses = jax.lax.scan(scan_fn, m_seq[0], m_seq[1:], length=k)
    return jnp.mean(losses)


def batched_rollout_loss(model, batch, k: int):
    # batch: m_seq (b, k+1, 3, nx, ny), H (b, 3)
    return jnp.mean(jax.vmap(lambda s, h: rollout_loss_fn(model, s, h, k))(
        batch["m_seq"], batch["H"]
    ))


@eqx.filter_jit(donate="all")
def update_fn(model, batch, optimizer, opt_state, k):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return batched_rollout_loss(eqx.combine(diff, static), batch, k)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state, params=diff)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


class RolloutSource(LLGStepperSource):
    """Length-(k+1) rollout windows, optionally expanded by the D4xZ2 symmetry.

    k=1 reduces to the one-step (m0, m1) source. With augment=True the index is
    multiplied by the 16 exact symmetry elements (see symmetry.py): each window
    is transformed on access, giving the model 16x more physically-exact samples
    spanning all field quadrants — the key to reaching SP4's OOD reversing field.
    """

    def __init__(self, path, k: int, augment: bool = False, max_workers: int = 16):
        self.trajs, self.fields = load_trajectories(path, max_workers)
        self.k = k
        self.n_groups = NUM_ELEMENTS if augment else 1
        self.index = np.array(
            [
                (ti, t, g)
                for ti, trj in enumerate(self.trajs)
                for t in range(trj.shape[0] - k)
                for g in range(self.n_groups)
            ],
            dtype=np.int64,
        )

    def __getitem__(self, idx):
        ti, t, g = self.index[idx]
        trj = self.trajs[ti]
        m_seq = trj[t : t + self.k + 1]
        H = self.fields[ti]
        if g != 0:
            m_seq = apply_field(m_seq, int(g))  # broadcasts over the time axis
            H = apply_H(H, int(g))
        return {"m_seq": m_seq, "H": H}


def make_schedule(lr, epochs, steps_per_epoch, warmup_frac=0.05):
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * warmup_frac))
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=lr * 0.02,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--learning-rate", type=float, required=True)
    ap.add_argument("--rollout-k", type=int, default=1, help="unroll length in loss")
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--augment", action="store_true", help="D4xZ2 symmetry aug (16x)")
    ap.add_argument("--eval-every", type=int, default=0, help="SP4 eval + keep best")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    key = jr.PRNGKey(args.seed)
    model_config = ModelConfig(
        hidden_channels=args.hidden_channels, num_blocks=args.num_blocks
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        size=args.size,
        checkpoint_every=10**9,
        cpu_buffer_size=8,
        device_buffer_size=4,
    )

    device = jax.devices()[0]
    train_src = RolloutSource(
        dataset_dir("train", args.size), k=args.rollout_k, augment=args.augment
    )
    train_loader = dataloader_factory(train_src, config=train_config, device=device)

    steps_per_epoch = max(1, len(train_src) // args.batch_size)
    key, subkey = jr.split(key)
    model = LLGEmulator(config=model_config, key=subkey)

    if args.cosine:
        lr = make_schedule(args.learning_rate, args.epochs, steps_per_epoch)
    else:
        lr = args.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=args.weight_decay),
    )
    opt_state = tx.init(eqx.filter(model, trainable_filter(model)))

    best_rmse, best_epoch = float("inf"), -1
    with tqdm(range(args.epochs)) as bar:
        for i in bar:
            losses = []
            for batch in train_loader(seed=i):
                model, opt_state, loss = update_fn(
                    model, batch, tx, opt_state, args.rollout_k
                )
                losses.append(loss)
            train_loss = jnp.stack(losses).mean().item()

            desc = f"train={train_loss:.4e}"
            # periodic SP4-rollout eval: keep the best-rollout checkpoint, not the
            # last (rollout-trained models can drift late). Cheap (~1s) vs an epoch.
            if args.eval_every and (i + 1) % args.eval_every == 0:
                from llg_emulator.bench import sp4_metrics

                rmse = sp4_metrics(model, args.size)["sp4_bulk_rmse"]
                desc += f" sp4={rmse:.4f}"
                if rmse < best_rmse:
                    best_rmse, best_epoch = rmse, i
                    save_model(model, model_config, Path(args.out))
            bar.set_description(desc)

    # If we were tracking best, the file already holds the best epoch; otherwise
    # (or if best never triggered) save the final model.
    if best_epoch < 0:
        save_model(model, model_config, Path(args.out))
        eval_model = model
    else:
        print(f"best sp4_bulk_rmse={best_rmse:.4f} at epoch {best_epoch}")
        eval_model = load_model(Path(args.out))

    metrics = evaluate(eval_model, args.size)
    for k_, v in metrics.items():
        print(f"{k_}: {v}")
    notes = args.notes or (
        f"k={args.rollout_k} lr={args.learning_rate} ep={args.epochs} "
        f"hc={args.hidden_channels} nb={args.num_blocks}"
        + (" cos" if args.cosine else "")
        + (" aug16" if args.augment else "")
        + (f" wd={args.weight_decay}" if args.weight_decay else "")
    )
    append_benchmark(args.tag, metrics, {"notes": notes})
    print(f"\nsaved {args.out}, appended '{args.tag}' to BENCHMARKS.md")


if __name__ == "__main__":
    main()
