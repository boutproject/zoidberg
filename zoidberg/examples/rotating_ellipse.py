import argparse

import numpy as np
from boututils.datafile import DataFile

import zoidberg as zb

from .common import (calc_curvilinear_curvature, calc_iota, get_lines,
                     smooth_metric)


def rotating_ellipse(
    nx=68,
    ny=16,
    nz=128,
    xcentre=5.5,
    I_coil=0.01,
    curvilinear=True,
    rectangular=False,
    fname="rotating-ellipse.fci.nc",
    a=0.4,
    curvilinear_inner_aligned=True,
    curvilinear_outer_aligned=True,
    npoints=2467,
    Btor=2.5,
    show_maps=False,
    calc_curvature=True,
    smooth_curvature=False,
    return_iota=True,
    write_iota=False,
):
    yperiod = 2 * np.pi / 5.0
    field = zb.field.RotatingEllipse(
        xcentre=xcentre, I_coil=I_coil, radius=2 * a, yperiod=yperiod, Btor=Btor
    )
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre + a / 2.0
    start_z = 0.0

    if rectangular:
        print("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(
            nx, nz, 1.0, 1.0, Rcentre=xcentre
        )
    elif curvilinear:
        print("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(
            R0=xcentre, a=a / 2.0, elong=0, triang=0.0, indent=0, n=npoints
        )
        outer = zb.rzline.shaped_line(
            R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints
        )

        if curvilinear_inner_aligned:
            print("Aligning to inner flux surface...")
            inner_lines = get_lines(
                field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
        if curvilinear_outer_aligned:
            print("Aligning to outer flux surface...")
            outer_lines = get_lines(
                field, xcentre + a, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )

        print("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner, outer in zip(inner_lines, outer_lines)
                ]
            else:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner in inner_lines
                ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)

    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)

    if curvilinear and calc_curvature:
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if calc_curvature and smooth_curvature:
        smooth_metric(
            fname, write_to_file=True, return_values=False, smooth_metric=True
        )

    if return_iota or write_iota:
        iota_bar = calc_iota(field, start_r, start_z)
        if write_iota:
            f = DataFile(str(fname), write=True)
            f.write("iota_bar", iota_bar)
            f.close()
        else:
            print("Iota_bar = ", iota_bar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", "-x", help="number of x points", default=68, type=int)
    parser.add_argument("--ny", "-y", help="number of y points", default=16, type=int)
    parser.add_argument("--nz", "-z", help="number of z points", default=128, type=int)
    parser.add_argument(
        "--xcentre", "-c", help="Centre of the screw pinch", default=5.5, type=float
    )
    parser.add_argument("--I-coil", "-i", help="Coil current", default=0.01, type=float)
    parser.add_argument("--curvilinear", help="???", action="store_false")
    parser.add_argument("--rectangular", help="???", action="store_true")
    parser.add_argument("--curvilinear_outer_aligned", help="???", action="store_false")
    parser.add_argument("--curvilinear_inner_aligned", help="???", action="store_false")
    parser.add_argument(
        "--fname",
        "-f",
        help="File name of the grid to be written",
        default="rotating-ellipse.fci.nc",
        type=str,
    )
    parser.add_argument(
        "--Btor", "-b", help="magnetic field on axis", default=2.5, type=float
    )
    parser.add_argument("-a", help="???", default=0.4, type=float)
    parser.add_argument(
        "--npoints",
        "-p",
        help="Number of iterations for surface tracing. Needs to be large enough.",
        default=2467,
        type=int,
    )

    parser.add_argument("--calc_curvature", help="???", action="store_false")
    parser.add_argument("--smooth_curvature", help="???", action="store_true")
    parser.add_argument(
        "--write_iota", help="Include iota in the grid file", action="store_true"
    )
    parser.add_argument("--show_maps", "-s", help="plot the grid", action="store_true")

    args = parser.parse_args()
    rotating_ellipse(return_iota=False, **args.__dict__)


if __name__ == "__main__":
    main()
