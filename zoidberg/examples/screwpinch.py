import argparse

import numpy as np
from boututils.boutarray import BoutArray

import zoidberg as zb

from .common import calc_curvilinear_curvature


def screwpinch(
    nx=68,
    ny=16,
    nz=128,
    xcentre=1.5,
    fname="screwpinch.fci.nc",
    a=0.2,
    npoints=421,
    show_maps=False,
):
    yperiod = 2 * np.pi
    field = zb.field.Screwpinch(xcentre=xcentre, yperiod=yperiod, shear=0)
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    print("Making curvilinear poloidal grid")
    inner = zb.rzline.shaped_line(
        R0=xcentre, a=a / 2.0, elong=0, triang=0.0, indent=0, n=npoints
    )
    outer = zb.rzline.shaped_line(
        R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints
    )
    poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)
    maps["y_coord"] = BoutArray(ycoords, dict(bout_type="ArrayY"))
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)
    calc_curvilinear_curvature(fname, field, grid, maps)

    if show_maps:
        zb.plot.plot_forward_map(grid, maps, yslice=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", "-x", help="number of x points", default=68, type=int)
    parser.add_argument("--ny", "-y", help="number of y points", default=16, type=int)
    parser.add_argument("--nz", "-z", help="number of z points", default=128, type=int)
    parser.add_argument(
        "--xcentre", "-c", help="Centre of the screw pinch", default=1.5, type=float
    )
    parser.add_argument(
        "--fname",
        "-f",
        help="File name of the grid to be written",
        default="screwpinch.fci.nc",
        type=str,
    )
    parser.add_argument("-a", help="???", default=0.2, type=float)
    parser.add_argument(
        "--npoints",
        "-p",
        help="Number of iterations for surface tracing. Needs to be large enough.",
        default=421,
        type=int,
    )

    parser.add_argument("--show_maps", "-s", help="plot the grid", action="store_true")

    args = parser.parse_args()
    screwpinch(**args.__dict__)


if __name__ == "__main__":
    main()
