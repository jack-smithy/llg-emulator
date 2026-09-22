import torch
from einops import rearrange
from torch import Tensor


def mse_loss(y, y_pred):
    return (y - y_pred).square().mean()


def prepare_batch(batch, device):
    x = batch["input_fields"]
    x = x.to(device)
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    y = batch["output_fields"]
    y = y.to(device)
    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

    return x, y


@torch.no_grad()
def rollout(model, x: Tensor, n_steps: int) -> Tensor:
    """Autoregressively predict n_steps frames from a context window.

    x: (B, Ti, Lx, Ly, F), as WellDataset serves input_fields.
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    n_fields = x.shape[-1]
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    frames = []
    for _ in range(n_steps):
        pred = model(x)
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
    for batch in loader:
        x, y = prepare_batch(batch, device)
        preds.append(model(x).cpu())
        truths.append(y.cpu())

    n_fields = truths[0].shape[1]  # n_steps_output=1, so channels == n_fields
    pattern = "B (T F) Lx Ly -> B T Lx Ly F"
    return (
        rearrange(torch.cat(preds), pattern, F=n_fields),
        rearrange(torch.cat(truths), pattern, F=n_fields),
    )
