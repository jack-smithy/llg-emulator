import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange, reduce
from jaxtyping import PyTree
from torch.utils.data import default_collate
from tqdm import tqdm

# `batch["constant_scalars"]` in the order the generator writes them
# (datagen.generate_varied_field.create_split); main.py checks the dataset agrees.
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")
MU_0 = 4e-7 * np.pi


def with_coords(m, grid):
    """Append the cell-centre coordinates in um as two channels.

    m: (B, 3, Lx, Ly); grid: (B, Lx, Ly, 2), `space_grid` as WellDataset serves
    it, i.e. the file's own (x, y) cell centres in metres. Taking it from the
    sample puts every geometry on the training set's scale.
    returns: (B, 5, Lx, Ly)
    """
    coords = rearrange(grid, "B Lx Ly D -> B D Lx Ly") * 1e6
    return jnp.concatenate((m, coords), axis=1)


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    return jtu.tree_map(np.asarray, default_collate(batch))


def conditioning(scalars, grid):
    """`constant_scalars` (B, 6) + `space_grid` (B, Lx, Ly, 2) -> `cond` (B, 3):
    `[Hx / Ms, Hy / Ms, log(delta / l_ex)]`.

    Hz is always zero in the data and Ms is constant, so the field components are
    the whole per-sample variation, nondimensionalised. `delta / l_ex` is the cell
    size over the exchange length `sqrt(2A / (mu0 Ms^2))` (~0.88 for the native
    5 nm cells), the one length ratio of the discretised problem; a log because
    it is a scale parameter.
    """
    h = scalars[:, 3:5] / scalars[:, :1]
    delta = grid[:, 1, 0, 0] - grid[:, 0, 0, 0]  # x-spacing; in-plane cells are square
    l_ex = jnp.sqrt(2 * scalars[:, 1] / (MU_0 * scalars[:, 0] ** 2))
    return jnp.concatenate([h, jnp.log(delta / l_ex)[:, None]], axis=1)


def downsample(m, grid, factor: int):
    """Coarse-grain m and grid onto the same film with `factor`-times larger cells.

    m: (B, 3, Lx, Ly) unit vectors; grid: (B, Lx, Ly, 2) cell centres in metres;
    Lx and Ly must be multiples of `factor`. Each coarse cell's m is the mean of
    its `factor` x `factor` block, renormalised back onto the unit sphere (lossy
    near domain walls, which is inherent to a larger cell); its centre is the
    block's mean centre.
    returns: (m, grid) on the coarse mesh.
    """
    if factor == 1:
        return m, grid
    m = reduce(m, "B F (x i) (y j) -> B F x y", "mean", i=factor, j=factor)
    grid = reduce(grid, "B (x i) (y j) D -> B x y D", "mean", i=factor, j=factor)
    return m / jnp.linalg.norm(m, axis=1, keepdims=True), grid


def prepare_batch(batch, pool: int = 1):
    """One-step batch -> `(m0, m1, cond)` in the model's layout, coarse-grained by
    `pool` first.

    m0: (B, 5, Lx, Ly), the frame with the cell grid appended; m1: (B, 3, Lx, Ly);
    cond: (B, 3).
    """
    m0 = rearrange(batch["input_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    m1 = rearrange(batch["output_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    m0, grid = downsample(m0, batch["space_grid"], pool)
    m1, _ = downsample(m1, batch["space_grid"], pool)
    c = conditioning(batch["constant_scalars"], grid)

    return with_coords(m0, grid), m1, c


@eqx.filter_jit
def predict(model, m0, cond):
    """One step for a batch: (B, 5, Lx, Ly), (B, 3) -> (B, 3, Lx, Ly)."""
    return jax.vmap(model)(m0, cond)


@eqx.filter_jit
def rollout(model, x, cond, grid, n_steps: int):
    """Autoregressively predict n_steps frames from one frame.

    x: (B, 1, Lx, Ly, F), as WellDataset serves input_fields.
    cond: (B, 3), as `conditioning` returns it.
    grid: (B, Lx, Ly, 2), as WellDataset serves space_grid.
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    m0 = rearrange(x, "B 1 Lx Ly F -> B F Lx Ly")

    def step(m, _):
        m = jax.vmap(model)(with_coords(m, grid), cond)
        return m, m

    _, frames = jax.lax.scan(step, m0, None, length=n_steps)
    return rearrange(frames, "T B F Lx Ly -> B T Lx Ly F")


def one_step_preds(model, loader):
    """Teacher-forced one-step predictions over a whole loader.

    returns: (pred, truth), both (N, 1, Lx, Ly, F) numpy -- the layout the
    well's metrics reduce over.
    """
    preds, truths = [], []
    for batch in tqdm(loader, mininterval=10):
        m0, m1, cond = prepare_batch(batch)
        preds.append(np.asarray(predict(model, m0, cond)))
        truths.append(m1)

    pattern = "B F Lx Ly -> B 1 Lx Ly F"
    return (
        rearrange(np.concatenate(preds), pattern),
        rearrange(np.concatenate(truths), pattern),
    )


def relative_norm_error(m):
    norm = np.linalg.norm(m, axis=-1)
    return norm.mean((1, 2))


def device_info():
    print(jax.devices())


@eqx.filter_jit
def loss_fn(model, m0, m1, cond):
    return jnp.mean(jnp.square(jax.vmap(model)(m0, cond) - m1))


@eqx.filter_jit(donate="all-except-first")
def update_fn(model: eqx.Module, batch: PyTree, optimizer, opt_state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, *batch)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
