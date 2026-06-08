from pdequinox.blocks import LinearChannelAdjustBlock
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from einops import rearrange, einsum
import jax


class TransolverAttention(eqx.Module):
    num_head: int
    temperature: float
    in_project_x: LinearChannelAdjustBlock
    in_project_fx: LinearChannelAdjustBlock
    in_project_slice: LinearChannelAdjustBlock
    q: LinearChannelAdjustBlock
    k: LinearChannelAdjustBlock
    v: LinearChannelAdjustBlock
    proj: LinearChannelAdjustBlock
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        num_spatial_dims,
        hidden_dim: int,
        num_heads: int,
        num_slices: int,
        p: float,
        temperature: float,
        key,
    ):
        dim_head = hidden_dim // num_heads
        self.num_head = num_heads
        self.temperature = temperature

        key, subkey = jr.split(key)

        self.in_project_x = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            use_bias=True,
            zero_bias_init=False,
            key=subkey,
        )

        self.in_project_fx = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            use_bias=True,
            zero_bias_init=False,
            key=subkey,
        )

        self.in_project_slice = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=dim_head,
            out_channels=num_slices,
            use_bias=True,
            zero_bias_init=False,
            key=subkey,
        )

        self.q = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=dim_head,
            out_channels=dim_head,
            use_bias=False,
            zero_bias_init=False,
            key=subkey,
        )

        self.k = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=dim_head,
            out_channels=dim_head,
            use_bias=False,
            zero_bias_init=False,
            key=subkey,
        )

        self.v = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=dim_head,
            out_channels=dim_head,
            use_bias=False,
            zero_bias_init=False,
            key=subkey,
        )

        self.proj = LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            use_bias=True,
            zero_bias_init=False,
            key=subkey,
        )

        self.dropout = eqx.nn.Dropout(p=p)

    def create_slices(self, x: Array):
        fx_mid = rearrange(
            self.in_project_fx(x),
            "n_points (n_heads dim_head) -> n_heads n_points dim_head",
        )

        x_mid = rearrange(
            self.in_project_x(x),
            "n_points (n_heads dim_head) -> n_heads n_points dim_head",
        )

        slice_weights = jax.nn.softmax(
            self.in_project_slice(x_mid) / self.temperature, axis=-1
        )

        slice_norm = rearrange(
            jnp.sum(slice_weights, axis=1),
            "num_heads num_slices -> num_heads num_slices 1",
        )

        slice_token = einsum(fx_mid, slice_weights, "hnc,hng->hgc") / (
            slice_norm + 1e-5
        )

        return slice_token, slice_weights

    def __call__(self, x: Array):
        slice_token, slice_weights = self.create_slices(x)

        q_slice_token = self.q(slice_token)
        k_slice_token = self.k(slice_token)
        v_slice_token = self.v(slice_token)

        out_slice_token = jax.nn.dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        )

        out_x = einsum(out_slice_token, slice_weights, "hgc,hng->hnc")
        out_x = rearrange(
            out_x, "num_heads num_points dim_head -> num_points (num_heads dim_head)"
        )
        return self.dropout(self.proj(out_x))


def main():
    key = jr.PRNGKey(0)

    model = TransolverAttention(
        num_spatial_dims=2,
        hidden_dim=16,
        num_heads=4,
        num_slices=4,
        p=0,
        temperature=0.5,
        key=key,
    )

    x = jnp.ones((16, 64, 64))

    y = model(x)

    print(y.shape)


if __name__ == "__main__":
    main()
