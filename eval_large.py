from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch
from einops import rearrange
from the_well.data import WellDataset
from torch import Tensor

from model import NormalizedFNO
from plot import plot_rollout
from utils import H_RANGE

dataset_path = "datasets/permalloy_varied_field"
device = "cuda"

IN_FRAMES = 1
F = 3


@torch.no_grad()
def rollout_means(
    model: NormalizedFNO,
    x: Tensor,
    meta: Tensor,
    n_steps: int,
) -> Tensor:
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
        pred_bulk = pred.mean((-1, -2))
        frames.append(pred_bulk)
        x = torch.cat([x[:, n_fields:], pred], dim=1)  # slide window one frame
    return torch.cat(frames, dim=0)


def main(seed, configuration):
    base_path = Path("results/permalloy_varied_field")
    results_path = base_path / configuration / f"seed_{seed}"
    print(results_path)

    model = NormalizedFNO(
        n_modes=(16, 16),
        in_channels=IN_FRAMES * F,
        out_channels=1 * F,
        hidden_channels=64,
        n_layers=2,
        norm="ada_in",
        ada_in_features=2,  # H = (Hx, Hy), Hz is always 0
    ).to(device)

    model.load_state_dict(torch.load(results_path / "model.pt", weights_only=False))
    print("model loaded")

    dataset = WellDataset(
        path=dataset_path,
        well_split_name="test",
        n_steps_input=100,
        n_steps_output=1,
        use_normalization=False,
    )
    print("dataset loaded")

    ref = dataset[0]["input_fields"].unsqueeze(0).to(device)
    h = dataset[0]["constant_scalars"][3:5].unsqueeze(0).to(device)
    h = (h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])

    print("starting rollout")
    start = perf_counter()
    pred_bulk = rollout_means(
        model, ref[:, :IN_FRAMES], h, n_steps=ref.shape[1] - IN_FRAMES
    )
    end = perf_counter()
    print(end - start)

    ref_bulk = ref.squeeze(0).mean(dim=(1, 2)).cpu().numpy()[IN_FRAMES:]
    pred_bulk = pred_bulk.cpu().numpy()

    print(ref_bulk.shape, pred_bulk.shape)

    fig, _ = plot_rollout(ref_bulk=ref_bulk, pred_bulk=pred_bulk)
    fig.savefig(results_path / "rollout_large.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", type=str, required=True)
    args = parser.parse_args()
    main(seed=args.seed, configuration=args.configuration)
