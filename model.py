from typing import Literal

import torch
import torch.linalg as LA
import torch.nn.functional as F
from neuralop.models import FNO
from torch import Tensor, nn

Number = float | int


class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer with flexible normalization."""

    embedding: Tensor | None = None

    def __init__(
        self,
        num_channels,
        meta_dim=1,
        norm_type="layer",
        num_groups=32,
        eps=1e-6,
        data_format="channels_first",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.meta_dim = meta_dim
        self.norm_type = norm_type.lower()
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (num_channels,)

        # Set up normalization layer based on type
        if self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=False)
        elif self.norm_type == "instance":
            self.norm = InstanceNorm()
        elif self.norm_type == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError(
                f"norm_type must be 'group', 'layer', 'instance', or 'identity', got {norm_type}"
            )

        # Map from meta_dim to channel affine parameters
        self.weight = nn.Linear(meta_dim, num_channels)
        self.bias = nn.Linear(meta_dim, num_channels)

    def _forward(self, x, meta=None):
        if self.norm_type in ["group", "instance"]:
            if self.data_format == "channels_last":
                x_for_norm = x.permute(0, -1, *range(1, x.dim() - 1))
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, *range(2, x.dim()), 1)
            else:
                x = self.norm(x)
        elif self.norm_type == "layer":
            if self.data_format == "channels_last":
                x = self.norm(x)
            else:
                x_for_norm = x.permute(0, *range(2, x.dim()), 1)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, -1, *range(1, x.dim() - 1))
        elif self.norm_type == "identity":
            x = self.norm(x)
        else:
            raise ValueError()

        if meta is None:
            return x

        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)

        meta = meta.type_as(x)
        weight = self.weight(meta)
        bias = self.bias(meta)

        if self.data_format == "channels_last":
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(1)
                bias = bias.unsqueeze(1)
            return weight * x + bias
        else:
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return weight * x + bias

    def set_embedding(self, embedding: Tensor) -> None:
        self.embedding = embedding

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x, self.embedding)


class NormalizedFNO(FNO):
    """An FNO for unit-norm vector fields, e.g. the magnetisation m.

    With `norm="ada_in"` every block's norm is a `FiLM` over `ada_in_features` per-sample
    scalars, e.g. the applied field, passed to `forward` as `meta`.

    The constructor mirrors `neuralop.models.FNO`'s: the signature is restated
    rather than swallowed into **kwargs because `BaseModel.__new__` inspects it
    to record the init kwargs a checkpoint is reloaded from, and because the
    upstream annotations declare several optional arguments non-optional.
    """

    def __init__(
        self,
        n_modes: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        positional_embedding: str | nn.Module | None = "grid",
        non_linearity: nn.Module = F.gelu,  # type: ignore
        norm: Literal["ada_in", "group_norm", "instance_norm"] | None = "ada_in",
        ada_in_features: int | None = None,
    ):
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,  # type: ignore
            non_linearity=non_linearity,
            norm=None if norm == "ada_in" else norm,  # type: ignore
        )
        if norm == "ada_in":
            # `FNO` does not pass `ada_in_features` on to `FNOBlocks`, and neuralop's `AdaIN`
            # holds one embedding for the whole batch; FiLM conditions each sample on its own.
            assert ada_in_features is not None
            self.fno_blocks.norm = nn.ModuleList(
                FiLM(
                    hidden_channels,
                    meta_dim=ada_in_features,
                    norm_type="instance",
                )
                for _ in range(n_layers * self.fno_blocks.n_norms)
            )

    # def forward(self, x: Tensor, output_shape=None, **kwargs) -> Tensor:
    #     dm = super().forward(x, output_shape=output_shape, **kwargs)  # (B, C, Lx, Ly)
    #     m_hat = x / (LA.norm(x, axis=1, keepdims=True) + 1e-8)  # (B, C, Lx, Ly)
    #     dm = dm - torch.sum(dm * m_hat, dim=0, keepdim=True) * m_hat
    #     m1 = x + dm
    #     return m1 / LA.norm(m1, dim=1, keepdims=True)

    def forward(  # type: ignore
        self,
        m0: Tensor,
        meta: Tensor | None = None,
        output_shape=None,
        **kwargs,
    ) -> Tensor:
        if meta is not None:
            self.fno_blocks.set_ada_in_embeddings(meta)
        m1 = super().forward(m0, output_shape=output_shape, **kwargs)  # (B, C, Lx, Ly)
        return m1 / LA.norm(m1, dim=1, keepdims=True)


def main():
    IN_CONTEXT_N = 1
    F = 3

    x = torch.ones((8, IN_CONTEXT_N * F, 64, 64))

    model = NormalizedFNO(
        n_modes=(16, 16),
        in_channels=IN_CONTEXT_N * F,
        out_channels=1 * F,
        hidden_channels=128,
        n_layers=5,
        norm="ada_in",
        ada_in_features=3,
    )

    y = model(x, meta=torch.zeros(8, 3))

    print(y.shape)


if __name__ == "__main__":
    main()
