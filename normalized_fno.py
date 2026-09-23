from typing import Literal

import torch
import torch.linalg as LA
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import FNO
from torch import Tensor, nn

import flowers

Number = float | int


class FiLM(flowers.FiLM):
    """`flowers.FiLM` behind neuralop's `AdaIN` interface.

    `FNOBlocks` calls its norms with the activations alone, so the conditioning is
    handed over beforehand through `set_embedding`, as `FNOBlocks.set_ada_in_embeddings`
    does for `AdaIN`. Unlike `AdaIN`, the embedding may differ per sample: (B, meta_dim).
    """

    embedding: Tensor | None = None

    def set_embedding(self, embedding: Tensor) -> None:
        self.embedding = embedding

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return super().forward(x, self.embedding)


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
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str | nn.Module | None = "grid",
        non_linearity: nn.Module = F.gelu,  # type: ignore
        norm: Literal["ada_in", "group_norm", "instance_norm"] | None = None,
        ada_in_features: int | None = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating"] | None = (
            "soft-gating"
        ),
        fno_skip: Literal["linear", "identity", "soft-gating"] | None = "linear",
        resolution_scaling_factor: Number | list[Number] | None = None,
        domain_padding: Number | list[Number] | None = None,
        fno_block_precision: Literal["full", "half", "mixed"] = "full",
        stabilizer: Literal["tanh"] | None = None,
        max_n_modes: tuple[int, ...] | None = None,
        factorization: Literal["Tucker", "CP", "TT"] | None = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: Literal["factorized", "reconstructed"] = "factorized",
        decomposition_kwargs: dict | None = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: type[nn.Module] = SpectralConv,
    ):
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,  # type: ignore
            non_linearity=non_linearity,
            norm=None if norm == "ada_in" else norm,  # type: ignore
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,  # type: ignore
            domain_padding=domain_padding,  # type: ignore
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,  # type: ignore
            max_n_modes=max_n_modes,  # type: ignore
            factorization=factorization,  # type: ignore
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,  # type: ignore
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,  # type: ignore
        )
        if norm == "ada_in":
            # `FNO` does not pass `ada_in_features` on to `FNOBlocks`, and neuralop's `AdaIN`
            # holds one embedding for the whole batch; FiLM conditions each sample on its own.
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
