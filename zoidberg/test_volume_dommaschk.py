import numpy as np
import xarray as xr
import sys
from . import field as zbfield, fieldtracer, rzline, zoidberg, poloidal_grid
import scipy
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

import os

script_dir = os.getcwd()


def calc_divertortheta(x, z, x0=0.0):
    theta = np.array(np.arctan2(z, x - x0))
    theta[theta < 0.0] += 2.0 * np.pi
    return theta


def dommaschk_grid_volume(nx, ny, nz, a1, a2, R0, Btor, symmetry, plotting=False):

    C = np.zeros((6, 3, 4))
    C[5, 2, 1] = -1.489
    C[5, 2, 2] = -1.489
    field = zbfield.DommaschkPotentials(C, R_0=R0, B_0=Btor)
    y_grid = np.linspace(0.0, yperiod, ny, endpoint=False)

    rzcoord, _ = fieldtracer.trace_poincare(
        field,
        (a1, a2),
        0.0,
        2.0 * np.pi / symmetry,
        y_slices=y_grid,
        revs=200,
        nplot=1,
        nover=20,
    )
    inner_lines = []
    outer_lines = []
    pol_grids = []
    for i in range(ny):

        inner_line = rzline.line_from_points(
            rzcoord[:, i, 0, 0], rzcoord[:, i, 0, 1], spline_order=1
        )
        outer_line = rzline.line_from_points(
            rzcoord[:, i, 1, 0], rzcoord[:, i, 1, 1], spline_order=1
        )

        inner_line = inner_line.equallySpaced(n=nz // 4)
        outer_line = outer_line.equallySpaced(n=nz // 4)

        inner_lines.append(inner_line)
        outer_lines.append(outer_line)

        pol_grid = poloidal_grid.grid_elliptic(
            inner_lines[i],
            outer_lines[i],
            nx,
            nz,
            restrict_size=2560,
            align=0,
            inner_ort=1,
            inner_maxmode=4,
            nx_inner=0,
            nx_outer=0,
        )
        pol_grids.append(pol_grid)
        if plotting:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
            pol_grid.plot(axis=ax, show=False)
            ax.set_aspect("equal")
            ax.grid(True, zorder=0)
            ax.set_xlim(1.6, 2.4)
            ax.set_ylim(-0.3, 0.3)
            plt.show()

    grid = grid.Grid(pol_grids, y_grid, 2.0 * np.pi / symmetry, yperiodic=True)
    maps = zoidberg.make_maps(grid, field, nslice=1, num=10)

    filename = os.path.join(script_dir, f"dommaschk_testgrid_{nx}_{ny}_{nz}.nc")
    zoidberg.write_maps(grid, field, maps, metric2d=False, gridfile=filename)
    gf = xr.open_dataset(filename)
    cellvolumes = (gf["J"] * gf["dx"] * gf["dy"] * gf["dz"]).values
    gridvolume = np.sum(cellvolumes)
    return gridvolume


def torus_volume_from_sections_periodic(polygons, angles, symmetry, mode=0):

    V = 0.0
    N = len(polygons)
    if mode == 0:
        for i in range(N):
            j = (i + 1) % N  # wrap around

            A1 = polygons[i].area
            A2 = polygons[j].area

            R1 = polygons[i].centroid.x
            R2 = polygons[j].centroid.x

            # Handle final interval correctly
            if j == 0:
                dphi = (2 * np.pi / symmetry - angles[i]) + angles[0]
            else:
                dphi = angles[j] - angles[i]

            V += 0.5 * (A1 * R1 + A2 * R2) * dphi

        return V
    elif mode == 1:
        dphi = np.mean(np.diff(angles))
        for i in range(N):
            R1 = polygons[i].centroid.x
            dl = 2.0 * np.pi * R1 * (dphi / (2.0 * np.pi))
            V += dl * polygons[i].area
        return V


def calculate_dommaschk_volume(
    ny, revs, symmetry, yperiod, plotting, Btor, R0, a1, a2, mode=1
):
    C = np.zeros((6, 3, 4))

    C[5, 2, 1] = -1.489
    C[5, 2, 2] = -1.489

    field = zbfield.DommaschkPotentials(C, R_0=R0, B_0=Btor)
    y_grid = np.linspace(0.0, yperiod, ny, endpoint=False)

    rzcoord, _ = fieldtracer.trace_poincare(
        field,
        (a1, a2),
        0.0,
        2.0 * np.pi / symmetry,
        y_slices=y_grid,
        revs=revs,
        nplot=1,
        nover=20,
    )

    inner_R = rzcoord[:, :, 0, 0]
    inner_Z = rzcoord[:, :, 0, 1]

    outer_R = rzcoord[:, :, -1, 0]
    outer_Z = rzcoord[:, :, -1, 1]

    inner_radius = np.sqrt((inner_R - R0) ** 2 + inner_Z**2)
    outer_radius = np.sqrt((outer_R - R0) ** 2 + outer_Z**2)
    inner_theta = calc_divertortheta(inner_R, inner_Z, R0)
    outer_theta = calc_divertortheta(outer_R, outer_Z, R0)

    inner_newR = np.zeros(inner_R.shape)
    inner_newZ = np.zeros(inner_R.shape)

    outer_newR = np.zeros(inner_R.shape)
    outer_newZ = np.zeros(inner_R.shape)

    newthetas = np.linspace(0.0, 2.0 * np.pi, revs, endpoint=False)

    cross_area = np.zeros(ny)
    all_inner_poly = []
    all_outer_poly = []
    for i in range(ny):
        interp_inner = scipy.interpolate.interp1d(
            inner_theta[:, i],
            inner_radius[:, i],
            fill_value="extrapolate",
            kind="linear",
        )
        interp_outer = scipy.interpolate.interp1d(
            outer_theta[:, i],
            outer_radius[:, i],
            fill_value="extrapolate",
            kind="linear",
        )

        inner_newradius = interp_inner(newthetas)
        outer_newradius = interp_outer(newthetas)

        inner_newR[:, i] = inner_newradius * np.cos(newthetas) + R0
        inner_newZ[:, i] = inner_newradius * np.sin(newthetas)

        outer_newR[:, i] = outer_newradius * np.cos(newthetas) + R0
        outer_newZ[:, i] = outer_newradius * np.sin(newthetas)

        inner_poly = Polygon(np.column_stack((inner_newR[:, i], inner_newZ[:, i])))
        outer_poly = Polygon(np.column_stack((outer_newR[:, i], outer_newZ[:, i])))

        all_inner_poly.append(inner_poly)
        all_outer_poly.append(outer_poly)

        cross_area[i] = outer_poly.area - inner_poly.area
        if plotting:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for poly in (inner_poly, outer_poly):
                x, y = poly.exterior.xy
                ax.plot(x, y, color="yellow")
            ax.scatter(
                inner_R[:, i], inner_Z[:, i], edgecolor="None", color="black", s=2.0
            )
            ax.scatter(
                outer_R[:, i], outer_Z[:, i], edgecolor="None", color="black", s=2.0
            )
            ax.scatter(
                inner_newR[:, i], inner_newZ[:, i], edgecolor="None", color="red", s=1.0
            )
            ax.scatter(
                outer_newR[:, i], outer_newZ[:, i], edgecolor="None", color="red", s=1.0
            )
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_ylim(-0.4, 0.4)
            ax.set_xlim(1.7, 2.3)
            plt.show()

    outer_volume = torus_volume_from_sections_periodic(
        all_outer_poly, y_grid, symmetry, mode=mode
    )
    inner_volume = torus_volume_from_sections_periodic(
        all_inner_poly, y_grid, symmetry, mode=mode
    )
    return outer_volume - inner_volume


# %%


def test_run():
    Btor = 2.5
    R0 = 2.0
    a1 = R0 + 0.03
    a2 = R0 + 0.08

    fail = False

    revs = 500
    symmetry = 5.0
    yperiod = 2.0 * np.pi / symmetry
    plotting = False

    # First do the exact calculation

    exact_volume = calculate_dommaschk_volume(
        2000, 500, symmetry, yperiod, plotting, Btor, R0, a1, a2, mode=1
    )
    print(f"Exact volume: {np.round(exact_volume,6)} m^3")

    scales = [2, 4, 8]
    gridvolumes = np.zeros(len(scales))

    for i in range(len(scales)):
        scale = scales[i]
        ny = 2 * scale
        nx = 8 * scale
        nz = 32 * scale
        gridvolumes[i] = dommaschk_grid_volume(
            nx, ny, nz, a1, a2, R0, Btor, symmetry, plotting=plotting
        )
        print(f"Grid volume: {np.round(gridvolumes[i],6)} m^3")
    error = np.abs(gridvolumes - exact_volume)
    coeffs = np.polyfit(np.log(scales), np.log(error), 1)
    a, b = coeffs

    if plotting:
        import matplotlib.pyplot as plt

        x = np.linspace(1.0, 50, 100)
        fig, ax = plt.subplots()
        ax.plot(scales, gridvolumes - exact_volume)
        ax.scatter(scales, np.abs(gridvolumes - exact_volume))
        ax.plot(x, 0.1 * np.power(x, -1), label=r"nx^-2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_xlabel(r"$n_x \propto n_y \propto n_z$")
        ax.set_ylabel(r"Error")
        plt.show()

    assert a <= -0.9


if __name__ == "__main__":
    test_run()
