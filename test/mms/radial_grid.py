import zoidberg as zb
import numpy as np
import xarray as xr
from mms_helper import lst

print(zb.__path__)
modes = [
    ("const", np.linspace),
    ("exp", np.geomspace),
]


def gen_name(*args):
    nx, ny, nz, R0, r0, r1, mode = args
    return f"radial_{modes[mode][0]}_{nx}_{ny}_{nz}_{R0}.fci.nc"


def gen_grid(nx, ny, nz, R0, r0, r1, mode=0):
    mode = modes[mode]
    one = np.ones((nx, ny, nz))
    r = mode[1](r0, r1, nx)[:, None]
    theta = np.linspace(0, 2 * np.pi, nz, False)[None, :]
    phi = np.linspace(0, 2 * np.pi / 5, ny, False)
    R = R0 + np.cos(theta) * r
    Z = np.sin(theta) * r
    pol_grid = zb.poloidal_grid.StructuredPoloidalGrid(R, Z)

    field = zb.field.CurvedSlab(Bz=0, Bzprime=0, Rmaj=R0)
    grid = zb.grid.Grid(pol_grid, phi, 5, yperiodic=True)

    fn = gen_name(*args)

    maps = zb.make_maps(grid, field, quiet=True, MXG=1)
    zb.write_maps(
        grid,
        field,
        maps,
        "tmp.nc",
        metric2d=False,
    )
    with xr.open_dataset("tmp.nc") as ds:
        dims = ds.dz.dims
        ds["r_minor"] = dims, one * r[:, None, :]
        ds["phi"] = "y", phi
        ds["theta"] = dims, one * theta[:, None, :]
        ds["one"] = dims, one
        ds.to_netcdf(fn)


def _togen(*args):
    return gen_name(*args), args


grids = {}
for mode in range(2):
    grids[modes[mode][0] + "1"] = [
        _togen(nz, 4, 4, 1, 0.1, 0.5, mode) + (nz,) for nz in lst[1:]
    ]
    grids[modes[mode][0] + "2"] = [
        _togen(nz, 4, 4, 2, 0.1, 1.1, mode) + (nz,) for nz in lst[1:]
    ]

if __name__ == "__main__":
    for todos in grids.values():
        for fn, args, _ in todos:
            gen_grid(*args)
