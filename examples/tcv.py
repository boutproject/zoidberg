# This script creates a structured TCV grid.

import json
import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray as xr

import zoidberg
import zoidberg as zb
from zoidberg.fieldtracer import trace_poincare

# The required data files are downloaded from the TCV-X21 GitHub repository:
# https://github.com/SPCData/TCV-X21
TCV_X21_BASE_URL = "https://raw.githubusercontent.com/SPCData/TCV-X21/main/"

# Files, relative to the TCV-X21 repository root, required by this example.
TCV_X21_REQUIRED_FILES = [
    "1.experimental_data/reference_scenario/65402_t1.eqdsk",
    "1.experimental_data/reference_scenario/divertor_polygon.json",
]


def download_tcv_data(datafolderpath):
    for relpath in TCV_X21_REQUIRED_FILES:
        outpath = os.path.join(datafolderpath, relpath)
        if not os.path.isfile(outpath):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            urlretrieve(TCV_X21_BASE_URL + relpath, outpath)


def rzphi_to_xyz(r, z, phi):
    return r * np.cos(phi), r * np.sin(phi), z


def cartesian_to_polar(x, z, x0=0.0):
    theta = np.array(np.arctan2(z, x - x0))
    theta[theta < 0.0] += 2.0 * np.pi

    radius = np.sqrt((x - x0) ** 2 + z**2)
    return radius, theta


def create_plate(x0, x1, z0, z1, n=50):
    factor = np.linspace(0.01, 0.99, n)
    newx = x0 + (x1 - x0) * factor
    newz = z0 + (z1 - z0) * factor
    result = np.zeros((n, 2))
    result[:, 0] = newx
    result[:, 1] = newz
    return result


class tcv_x21:
    def __init__(
        self,
        tcv_datafolderpath=None,
        tcv_magneticfieldpath="1.experimental_data/reference_scenario/"
        + "65402_t1.eqdsk",
        tcv_divertordatapath="1.experimental_data/reference_scenario/divertor_polygon.json",
    ):
        # If no path to an existing TCV-X21 checkout is given, download the
        # required files first.
        self.datafolderpath = tcv_datafolderpath or "tcv_data"
        download_tcv_data(self.datafolderpath)
        self.magneticfieldpath = tcv_magneticfieldpath
        self.magnetic_field = zoidberg.field.GEQDSK(
            os.path.join(self.datafolderpath, self.magneticfieldpath)
        )
        self.divertordatapath = tcv_divertordatapath

        with open(
            os.path.join(self.datafolderpath, self.divertordatapath), "r"
        ) as file:
            self.divertordata = json.load(file)

    def set_value(self, optdir, setting, value):
        optdir[str(setting)] = value
        setattr(self, setting, value)

    def plot_divertorplates(self, ax, c="black", s=0.1):
        return ax.plot(
            self.divertordata["r_points"],
            self.divertordata["z_points"],
            color=c,
            linewidth=s,
        )

    def divertor(
        self,
        divertoroption,
        nx,
        ny,
        nz,
        x1,
        x2,
        y_cutoff,
        x_cutoff,
        ddd_factor_div=2,
        n_plate=10,
        plotting=False,
        tracemultiplier=1.0,
        equalize_J=0,
    ):
        self.divertoropt = {}
        params = {k: v for k, v in locals().items() if k != "self"}
        for key, value in params.items():
            self.set_value(self.divertoropt, key, value)

        ddd = self.nz * self.ddd_factor_div

        if divertoroption == 1:
            rzcoord, ycoords = trace_poincare(
                self.magnetic_field,
                np.linspace(0.65, 1.0, 100),
                self.y_cutoff,
                2.0 * np.pi / ddd,
                y_slices=[0],
                revs=int(tracemultiplier * ddd),
                nplot=1,
            )
            chosen_rzcoord, chosen_ycoord = trace_poincare(
                self.magnetic_field,
                x1,
                self.y_cutoff,
                2.0 * np.pi / ddd,
                y_slices=[0],
                revs=int(tracemultiplier * ddd),
                nplot=1,
            )
            chosen_rzcoord2, chosen_ycoord2 = trace_poincare(
                self.magnetic_field,
                x2,
                self.y_cutoff,
                2.0 * np.pi / ddd,
                y_slices=[0],
                revs=int(tracemultiplier * ddd),
                nplot=1,
            )

            self.poincare_RZ = rzcoord
            bool_outer = chosen_rzcoord[:, :, :, 0] > self.x_cutoff
            bool_inner = chosen_rzcoord2[:, :, :, 0] > self.x_cutoff
            chosen_rzcoord = chosen_rzcoord[bool_outer]
            chosen_rzcoord2 = chosen_rzcoord2[bool_inner]

            plate1 = create_plate(
                self.x1, self.x2, self.y_cutoff, self.y_cutoff, n=self.n_plate
            )
            plate2 = create_plate(
                self.x_cutoff,
                self.x_cutoff,
                chosen_rzcoord2[-1, 1],
                chosen_rzcoord[-1, 1],
                n=self.n_plate,
            )

            outer_line = np.concatenate(
                (chosen_rzcoord2, plate2, chosen_rzcoord[::-1], plate1), axis=0
            )
            outer_line = zoidberg.rzline.RZline(
                outer_line[:, 0], outer_line[:, 1], smooth=0
            )
            outer_line = outer_line.equallySpaced(n=self.nz)
            from zoidberg.smooth import smooth as sm

            newx, newz = sm(
                np.roll(outer_line.R, -30),
                np.roll(outer_line.Z, -30),
                cutoff=0.98,
                bounds=self.nz // 128,
                bounds_find=self.nz // 128,
                plot=False,
            )

            outer_line = zoidberg.rzline.RZline(newx, newz, smooth=0)
            outer_line = outer_line.equallySpaced(n=self.nz)
            newx = outer_line.R
            newz = outer_line.Z

            self.outer_line = zoidberg.rzline.RZline(newx, newz, smooth=0)

            if plotting:
                _, ax = plt.subplots()
                for i in range(np.shape(rzcoord)[0]):
                    ax.scatter(
                        rzcoord[i, 0, :, 0],
                        rzcoord[i, 0, :, 1],
                        s=0.5,
                        edgecolors="None",
                        c="black",
                    )

                for i in range(chosen_rzcoord.shape[0]):
                    ax.scatter(
                        chosen_rzcoord[i, 0],
                        chosen_rzcoord[i, 1],
                        s=1.5,
                        edgecolors="None",
                        c="red",
                    )
                for i in range(chosen_rzcoord2.shape[0]):
                    ax.scatter(
                        chosen_rzcoord2[i, 0],
                        chosen_rzcoord2[i, 1],
                        s=1.5,
                        edgecolors="None",
                        c="red",
                    )

                ax.plot(self.outer_line.R, self.outer_line.Z, color="orange")
                ax.set_xlim(0.5, 1.2)
                ax.set_ylim(-0.8, 0.6)
                ax.set_aspect("equal")
                ax.scatter(
                    self.divertordata["r_points"],
                    self.divertordata["z_points"],
                    c="green",
                    edgecolors="None",
                    s=1.0,
                )
                plt.show()

    def core(self, inner_fluxsurfaceposition, ddd_factor_core=2, Rmaj=0.9):

        self.coreopt = {}
        params = {k: v for k, v in locals().items() if k != "self"}
        for key, value in params.items():
            self.set_value(self.coreopt, key, value)

        self.Rmaj = Rmaj

        ddd = self.nz * ddd_factor_core
        chosen_rzcoord3, chosen_ycoord3 = trace_poincare(
            self.magnetic_field,
            self.inner_fluxsurfaceposition,
            0.0,
            2.0 * np.pi / ddd,
            y_slices=[0],
            revs=int(ddd),
            nplot=1,
        )
        chosen_rzcoord3 = chosen_rzcoord3[:, 0, 0, :]

        inner_line = zoidberg.rzline.line_from_points(
            chosen_rzcoord3[:, 0], chosen_rzcoord3[:, 1], spline_order=1
        )
        self.inner_line = inner_line.equallySpaced(n=self.nz)

    def plot_inner_outer(self, offset=0.02, plot_plates=False):
        _, ax = plt.subplots(layout="constrained")
        ax.plot(self.outer_line.R, self.outer_line.Z)
        ax.plot(self.inner_line.R, self.inner_line.Z)
        for i in range(np.shape(self.poincare_RZ)[0]):
            ax.scatter(
                self.poincare_RZ[i, 0, :, 0],
                self.poincare_RZ[i, 0, :, 1],
                s=0.5,
                edgecolors="None",
                c="black",
            )
        ax.set_xlim(
            np.min((self.outer_line.R, self.inner_line.R)) - offset,
            np.max((self.outer_line.R, self.inner_line.R)) + offset,
        )
        ax.set_ylim(
            np.min((self.outer_line.Z, self.inner_line.Z)) - offset,
            np.max((self.outer_line.Z, self.inner_line.Z)) + offset,
        )
        ax.set_aspect("equal")
        if plot_plates:
            self.plot_divertorplates(ax, s=0.5, c="green")
        plt.show()

    def create_grids(
        self,
        restrict_size,
        align,
        inner_ort,
        inner_maxmode,
        nx_inner,
        nx_outer,
        slab_diff=0.02,
        dz_relax=None,
        redistribute=False,
    ):
        self.gridopt = {}
        params = {k: v for k, v in locals().items() if k != "self"}
        for key, value in params.items():
            self.set_value(self.gridopt, key, value)

        self.inner_pointnum = self.nz

        pol_grids = []
        if self.divertoroption == 1:
            pol_grid = zoidberg.poloidal_grid.grid_elliptic(
                self.inner_line,
                self.outer_line,
                self.nx,
                self.nz,
                restrict_size=self.restrict_size,
                align=self.align,
                inner_ort=self.inner_ort,
                inner_maxmode=self.inner_maxmode,
                nx_inner=self.nx_inner,
                nx_outer=self.nx_outer,
                dz_relax=self.dz_relax,
            )

        for i in range(self.ny):
            pol_grids.append(pol_grid)
        self.pol_grids = pol_grids

    def plot_grid(self, show=True):
        fig, ax = plt.subplots(layout="constrained")
        self.pol_grids[0].plot(axis=ax, show=False)
        ax.set_aspect("equal")
        if show:
            plt.show()
        else:
            return fig, ax

    def write_grids(
        self,
        symmetry,
        axysymmetric,
        write_folder,
        nslice=1,
        use_outside=False,
        return_maps=False,
        n_chunks=8,
    ):
        self.writeopt = {}
        params = {k: v for k, v in locals().items() if k != "self"}
        for key, value in params.items():
            self.set_value(self.writeopt, key, value)

        ycoords = np.linspace(0, 2.0 * np.pi / self.symmetry, self.ny, endpoint=False)

        self.grid = zb.grid.Grid(
            self.pol_grids, ycoords, 2.0 * np.pi / self.symmetry, yperiodic=True
        )

        self.maps = zb.make_maps(
            self.grid, self.magnetic_field, nslice=self.nslice, n_chunks=n_chunks
        )

        if return_maps:
            return self.maps

        # Bend the most inner lines back on the flux surface
        if self.divertoroption == 1:
            self.maps["forward_xt_prime"][2, :, :] = 2.0
            self.maps["backward_xt_prime"][2, :, :] = 2.0

        self.fn = f"TCV_X21_{self.nx}_{self.ny}_{self.nz}_{self.symmetry}_{self.divertoroption}.fci.grid.nc"

        filename = os.path.join(self.write_folder, self.fn)
        zoidberg.write_maps(
            self.grid, self.magnetic_field, self.maps, metric2d=False, gridfile=filename
        )
        # dagp(filename)
        with netCDF4.Dataset(filename, mode="r+") as nc:
            nc.setncattr("gridopt", json.dumps(self.gridopt))
            nc.setncattr("coreopt", json.dumps(self.coreopt))
            nc.setncattr("divertoropt", json.dumps(self.divertoropt))
        self.filename = filename

    def correct_grids(self, write_corrected=False):

        gridcopy = xr.open_dataset(self.filename)
        a = (gridcopy.forward_xt_prime) > (gridcopy.nx - 3)
        b = (gridcopy.backward_xt_prime) > (gridcopy.nx - 3)

        indxa = np.where(a)[0]
        indya = np.where(a)[1]
        indza = np.where(a)[2]
        indxb = np.where(b)[0]
        indyb = np.where(b)[1]
        indzb = np.where(b)[2]

        thisy = 0
        plt.close("all")
        _, (ax1) = plt.subplots(1, 1, figsize=(4.8, 4.8), layout="constrained")
        am = ax1.pcolormesh(
            gridcopy.R.isel(y=thisy), gridcopy.Z.isel(y=thisy), (a + b)[:, thisy, :]
        )
        plt.colorbar(am, label="cells outside xout")
        plt.show()

        _, (ax1) = plt.subplots(1, 1, figsize=(4.8, 4.8), layout="constrained")
        am = ax1.pcolormesh(
            gridcopy.R.isel(y=thisy),
            gridcopy.Z.isel(y=thisy),
            gridcopy.forward_xt_prime[:, thisy, :],
        )
        plt.colorbar(am, label="foward_xt_prime")
        plt.show()

        _, (ax1) = plt.subplots(1, 1, figsize=(4.8, 4.8), layout="constrained")
        am = ax1.pcolormesh(
            gridcopy.R.isel(y=thisy),
            gridcopy.Z.isel(y=thisy),
            gridcopy.backward_xt_prime[:, thisy, :],
        )
        plt.colorbar(am, label="backward_xt_prime")
        plt.show()

        a = (gridcopy.forward_xt_prime) < 0.0
        b = (gridcopy.backward_xt_prime) < 0.0
        indxa = np.where(a)[0]
        indya = np.where(a)[1]
        indza = np.where(a)[2]
        indxb = np.where(b)[0]
        indyb = np.where(b)[1]
        indzb = np.where(b)[2]

        _, ax = plt.subplots()
        ax.pcolormesh(
            gridcopy.R.isel(y=thisy),
            gridcopy.Z.isel(y=thisy),
            gridcopy.forward_xt_prime[:, thisy, :],
        )
        for i in range(indxa.size):
            if indya[i] == 0:
                ax.plot(
                    (
                        gridcopy.R[indxa[i], indya[i], indza[i]],
                        gridcopy.forward_R[indxa[i], indya[i], indza[i]],
                    ),
                    (
                        gridcopy.Z[indxa[i], indya[i], indza[i]],
                        gridcopy.forward_Z[indxa[i], indya[i], indza[i]],
                    ),
                )
        ax.set_aspect("equal")
        plt.show()

        _, ax = plt.subplots()
        ax.pcolormesh(
            gridcopy.R.isel(y=thisy),
            gridcopy.Z.isel(y=thisy),
            gridcopy.backward_xt_prime[:, thisy, :],
        )
        for i in range(indxa.size):
            if indya[i] == 0:
                ax.plot(
                    (
                        gridcopy.R[indxa[i], indya[i], indza[i]],
                        gridcopy.backward_R[indxa[i], indya[i], indza[i]],
                    ),
                    (
                        gridcopy.Z[indxa[i], indya[i], indza[i]],
                        gridcopy.backward_Z[indxa[i], indya[i], indza[i]],
                    ),
                )
        ax.set_aspect("equal")
        plt.show()

        if write_corrected:
            if self.divertoroption == 1:
                for i in range(indxa.size):
                    if indxa[i] > 5:
                        gridcopy.forward_xt_prime[indxa[i], indya[i], indza[i]] = (
                            self.nx
                        )

                for i in range(indxb.size):
                    if indxb[i] > 5:
                        gridcopy.backward_xt_prime[indxb[i], indyb[i], indzb[i]] = (
                            self.nx
                        )

            newfn = os.path.join(self.write_folder, "cor_" + self.fn)
            self.newfilename = newfn
            gridcopy.to_netcdf(newfn)
        return gridcopy


# %%


# Multiplication factor for the resolution.
fac = 0.5

# Divertor option. In this script just one.
divoption = 1

# Symmetry of the system. We just simulate one 16th of the device here.
symmetry = 16


# Lower plate, inner radial position
x1 = 0.74

# Lower plate, outer radial position
x2 = 0.865

# We assume the divertor is at -0.7, not at -0.75.
y_cutoff = -0.70

# Inner Wall is at 0.625, we assume it is at 0.62
x_cutoff = 0.62

# tcv_datafolderpath defaults to ./tcv_data. If the required files are not
# present, they will be downloaded automatically from the TCV-X21 GitHub repo
# (https://github.com/SPCData/TCV-X21). To use a local checkout instead,
# pass its path: tcv = tcv_x21(tcv_datafolderpath="/path/to/TCV-X21/")
tcv = tcv_x21()


tcv.divertor(
    divoption,
    nx=int(fac * 128 + 4),
    ny=2,
    nz=int(1024 * fac),
    x1=x1,
    x2=x2,
    y_cutoff=y_cutoff,
    x_cutoff=x_cutoff,
    plotting=0,
    tracemultiplier=2,
)


# Outboard midplane is at 1.09 here.
corepos = 1.09

tcv.core(corepos)
tcv.plot_inner_outer(plot_plates=True)


# %%

tcv.create_grids(
    restrict_size=2560,
    align=False,
    inner_ort=True,
    inner_maxmode=4,
    nx_inner=2,
    nx_outer=0,
    dz_relax=1.75,
    redistribute=False,
    slab_diff=0.05,
)  # 1.75

_, ax = tcv.plot_grid(show=False)
ax.plot(tcv.outer_line.R, tcv.outer_line.Z)
plt.show()


# %% Create the poloidal grids and create the maps

# write_folder determines where the grid will be saved.
# By default the current working directory is used.
write_folder = os.getcwd() if len(sys.argv) < 2 else sys.argv[1]

tcv.write_grids(
    symmetry=symmetry,
    axysymmetric=False,
    write_folder=write_folder,
    use_outside=False,
    return_maps=False,
)
gc = tcv.correct_grids(write_corrected=True)

_, ax = plt.subplots()
for i in range(5):
    plt.plot(gc.backward_xt_prime[i, 0, :], label=str(i))
plt.legend()
plt.show()
