import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from loader import JaxLoader, LLGDataset, to_device
from model import LLGEmulator

jax.config.update("jax_compilation_cache_dir", ".jax_cache")


def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


@eqx.filter_jit
def loss_fn(model, m0, m1):
    m1_pred = jax.vmap(model)(m0)
    return jnp.mean(jnp.square(m1 - m1_pred))


@eqx.filter_jit
def update_fn(model, m0, m1, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, m0, m1)
    updates, state = optimizer.update(grad, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss


def main():
    train_dataset = LLGDataset(
        "../micromagnetic-data/data/v2/low_res/train",
        warmup_steps=1,
    )
    val_dataset = LLGDataset(
        "../micromagnetic-data/data/v2/low_res/val",
        warmup_steps=1,
    )

    train_loader = JaxLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = JaxLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )

    train_ratio = len(train_dataset) / (len(val_dataset) + len(train_dataset))
    print(f"num train samples = {len(train_loader)}")
    print(f"num val samples = {len(val_loader)}")
    print(f"train ratio = {train_ratio * 100:.2f}%")

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)

    # target device for single-device training
    device = jax.devices()[1]

    model = LLGEmulator(
        num_modes=32,
        num_blocks=4,
        hidden_channels=64,
        key=subkey,
    )
    dynamic, static = eqx.partition(model, eqx.is_array)
    dynamic = to_device(dynamic, device=device)
    model = eqx.combine(dynamic, static)

    print(f"num parameters = {count_parameters(model)}")

    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_history = []
    val_history = []
    with tqdm(range(64)) as bar:
        for i in bar:
            epoch_train_loss = 0
            for batch in train_loader:
                batch = to_device(batch, device=device)
                model, state, loss = update_fn(
                    model,
                    *batch,
                    optimizer=optimizer,
                    state=state,
                )
                epoch_train_loss += loss.item()
            train_history.append(epoch_train_loss / len(train_loader))

            epoch_val_loss = 0
            for batch in val_loader:
                batch = to_device(batch, device=device)
                loss = loss_fn(model, *batch)
                epoch_val_loss += loss.item()
            bar.set_description(f"loss={epoch_val_loss / len(val_loader):.4e}")
            val_history.append(epoch_val_loss / len(val_loader))
            eqx.tree_serialise_leaves(f"weights_epoch_{i}.eqx", model)

    eqx.tree_serialise_leaves("weights_alt.eqx", model)

    plt.semilogy(train_history, label="train")
    plt.semilogy(val_history, label="val")
    plt.legend()
    plt.savefig("learning_curve_alt.png")


if __name__ == "__main__":
    main()
