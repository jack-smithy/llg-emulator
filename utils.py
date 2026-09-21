import numpy as np
from einops import rearrange
from torch import Tensor


def mse_loss(y: Tensor, y_pred: Tensor) -> Tensor:
    return (y - y_pred).square().mean()


def prepare_batch(batch, device):
    x = batch["input_fields"]
    x = x.to(device)
    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

    y = batch["output_fields"]
    y = y.to(device)
    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

    return x, y


def train_epoch(loader, model, optimizer, device, loss_fn):
    history = []
    for batch in loader:
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

        fx = model(x)

        mse = loss_fn(y, fx)
        mse.backward()

        optimizer.step()
        optimizer.zero_grad()

        history.append(mse.item())

    return np.mean(history)


def val_epoch(loader, model, device, loss_fn):
    history = []
    for batch in loader:
        x = batch["input_fields"]
        x = x.to(device)
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

        y = batch["output_fields"]
        y = y.to(device)
        y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

        fx = model(x)

        mse = loss_fn(y, fx)

        history.append(mse.item())

    return np.mean(history)
