import equinox as eqx
import pdequinox as pdeqx
import jax.numpy.linalg as LA


class ResidualEmulator(eqx.Module):
    network: eqx.Module

    def __init__(self, network, normalization_factor: float = 1.0):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(
            network=network, normalization_factor=normalization_factor
        )

    def __call__(self, m0, meta_data):
        dm = self.network(m0, meta_data=meta_data)  # type: ignore
        m1 = m0[:3] + dm
        return m1 / LA.norm(m1, axis=0, keepdims=True)
