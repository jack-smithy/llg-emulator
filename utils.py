import torch
from einops import rearrange
from torch import Tensor
import numpy as np
from tqdm import tqdm
import torch.linalg as LA

MU_0 = 4 * torch.pi * 1e-7

H_RANGE = (-50e-3 / MU_0, 50e-3 / MU_0)


def mse_loss(y, y_pred):
    return (y - y_pred).square().mean()


def prepare_batch(batch, device):
    x = batch["input_fields"]
    x = x.to(device)
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    y = batch["output_fields"]
    y = y.to(device)
    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

    # the applied field the model is conditioned on, in units of the train split's rms |H|
    h = batch["constant_scalars"][:, 3:5].to(device)
    h = (h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])
    return x, y, h


def prepare_batch_h_field(batch, device):
    x = batch["input_fields"]
    x = x.to(device)
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    y = batch["output_fields"]
    y = y.to(device)
    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

    h_scalars = batch["constant_scalars"][:, 3:5].to(device)
    h_scalars = (h_scalars - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])

    B, _, *L = x.shape
    h = torch.ones((B, 2, *L), device=device)
    h[:, 0] *= h_scalars[0]
    h[:, 1] *= h_scalars[1]

    return x, y, h


@torch.no_grad()
def rollout(model, x: Tensor, meta: Tensor, n_steps: int) -> Tensor:
    """Autoregressively predict n_steps frames from a context window.

    x: (B, Ti, Lx, Ly, F), as WellDataset serves input_fields.
    meta: (B, 3), the scaled applied field, as prepare_batch returns it.
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    n_fields = x.shape[-1]
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    frames = []
    for _ in range(n_steps):
        pred = model(x, meta)
        frames.append(pred)
        x = torch.cat([x[:, n_fields:], pred], dim=1)  # slide window one frame

    return rearrange(
        torch.cat(frames, dim=1), "B (T F) Lx Ly -> B T Lx Ly F", F=n_fields
    )


@torch.no_grad()
def one_step_preds(model, loader, device):
    """Teacher-forced one-step predictions over a whole loader.

    returns: (pred, truth), both (N, 1, Lx, Ly, F) on cpu -- the layout the
    well's metrics reduce over.
    """
    preds, truths = [], []
    for batch in tqdm(loader):
        x, y, h = prepare_batch(batch, device)
        preds.append(model(x, h).cpu())
        truths.append(y.cpu())

    n_fields = truths[0].shape[1]  # n_steps_output=1, so channels == n_fields
    pattern = "B (T F) Lx Ly -> B T Lx Ly F"
    return (
        rearrange(torch.cat(preds), pattern, F=n_fields),
        rearrange(torch.cat(truths), pattern, F=n_fields),
    )


def relative_norm_error(m):
    norm = np.linalg.norm(m, axis=-1)
    return norm.mean((1, 2))


def device_info():
    try:
        print(torch.cuda.get_device_name(0))
    except RuntimeError:
        print("no gpus found")


def normalize(m: Tensor) -> Tensor:
    assert len(m.shape) == 4  # (B, C, W, H)
    return m / LA.norm(m, dim=1, keepdims=True)


def train_epoch(model, loader, loss_fn, optimizer, device) -> float:
    running_loss = 0
    model.train()
    for batch in tqdm(loader, mininterval=10):
        x, y, h = prepare_batch(batch, device)

        fx = model(x, meta=h)
        loss = loss_fn(y, fx)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    return running_loss / len(loader)


def val_epoch(model, loader, loss_fn, device) -> float:
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, mininterval=10):
            x, y, h = prepare_batch(batch, device)

            fx = model(x, meta=h)
            loss = loss_fn(y, fx)

            running_loss += loss.item()
    return running_loss / len(loader)
