"""Coarse-graining by block averaging.

The stored magnetisation is the moment density in units of Ms, i.e. the volume fraction times
the unit magnetisation (0 in air). Its block average is the coarse cell's moment density, so the
total moment is conserved and |m| < 1 wherever a block holds walls, domains or air.
"""

import torch
import torch.nn.functional as F


def block_average(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Average cubes of factor^3 cells. x: (nx, ny, nz, channels) or (batch, nx, ny, nz, channels).

    Each of nx, ny, nz must be divisible by factor (avg_pool3d would silently drop the rest).
    Averaging by 2 and then by 2 again equals averaging by 4, so every coarse level can be made
    from the stored one.
    """
    if any(n % factor for n in x.shape[-4:-1]):
        raise ValueError(f"grid {tuple(x.shape[-4:-1])} is not divisible by {factor}")
    return F.avg_pool3d(x.movedim(-1, -4), factor).movedim(-4, -1)
