import torch.nn as nn
from torch import Tensor


class LearnedField(nn.Module):
    """h_theta[m] = Ms CNN(m), an extra effective field output directly, with no energy behind it."""

    def __init__(self, Ms):
        super().__init__()
        # model.WallEnergy's CNN with 3 output channels, the field's x, y, z components, so a
        # comparison with WallEnergy is only about energy vs direct field
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            # no bias: a field that does not depend on m would act as a second applied field
            nn.Conv3d(32, 3, kernel_size=1, bias=False),
        )
        # h_theta starts at 0, so training starts from exchange + demag + applied field
        nn.init.zeros_(self.cnn[-1].weight)
        self.Ms = Ms

    def field(self, m: Tensor) -> Tensor:
        """m (B, X, Y, Z, 3) -> h_theta (B, X, Y, Z, 3) [A/m]."""
        h = self.cnn(m.permute(0, 4, 1, 2, 3))  # (B, 3, X, Y, Z), in units of Ms
        return self.Ms * h.permute(0, 2, 3, 4, 1)
