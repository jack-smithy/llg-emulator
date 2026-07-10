import numpy as np
from pyevtk.hl import gridToVTK


def write_vtr(field, filename, n, dx):
    # save result
    x = np.arange(0, (0.1 + n[0]) * dx[0], dx[0], dtype="float64")
    y = np.arange(0, (0.1 + n[1]) * dx[1], dx[1], dtype="float64")
    z = np.arange(0, (0.1 + n[2]) * dx[2], dx[2], dtype="float64")

    # scalar data
    if len(field.shape) == 3:
        gridToVTK(filename, x, y, z, cellData={"f": field.copy()})
    else:
        gridToVTK(
            filename,
            x,
            y,
            z,
            cellData={
                "f": (
                    field[:, :, :, 0].copy(),
                    field[:, :, :, 1].copy(),
                    field[:, :, :, 2].copy(),
                )
            },
        )
