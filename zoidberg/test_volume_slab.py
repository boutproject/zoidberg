import os

import numpy as np
import scipy
import xarray as xr
from shapely.geometry import Polygon
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/IPP-HGW/toto/Dokumente/my-zoidberg/zoidberg")
import zoidberg

from zoidberg import field as zbfield
from zoidberg import fieldtracer
from zoidberg import grid as zbgrid
from zoidberg import poloidal_grid, rzline, zoidberg

script_dir = os.getcwd()


def polyfunc(y, p):
    result = p[0]
    for i in range(1, len(p)):
        result += p[i] * np.sin(i * y)
    return result


class testfield(zbfield.MagneticField):
    def __init__(self, polynomials=[1.0]):
        """
        Parameters
        ----------
        polynomials : List of floats
            Sinus coefficients for the magnetic field strength:
            B(phi) = polynomials[0] + polynomials[1]*sin(phi) + polynomials[2]*sin(2.0 * phi) ...

        """
        self.polynomials = polynomials
        self.Bz = 0.0
        self.Bzprime = 0.0
        self.xcentre = 5.0

    def Byfunc(self, x, z, phi):
        return polyfunc(phi, self.polynomials)

    def Bxfunc(self, x, z, phi):
        return np.zeros(x.shape)

    def Bzfunc(self, x, z, phi):
        return np.zeros(x.shape)


# %% Create a slab grid and calculate the volumes of the cells
field = testfield([1.0, 0.3, -0.5, 0.2])
nx = 20
ny = 8
nz = 20
pol_grids = []
for i in range(ny):
    pol_grids.append(
        poloidal_grid.RectangularPoloidalGrid(
            nx=nx, nz=nz, Lx=0.05, Lz=0.05, Rcentre=5.0
        )
    )

grid = zbgrid.Grid(
    pol_grids,
    np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False),
    2.0 * np.pi,
    yperiodic=True,
)
reffactors = [1, 5, 10, 20, 50]

maps = {}
for ref in reffactors:
    maps[str(ref)] = zoidberg.make_maps(grid, field, refine_parallel_integral=ref)


fn = f"testslabgrid_{nx}_{ny}_{nz}.nc"
filename = os.path.join(script_dir, fn)
zoidberg.write_maps(grid, field, maps["1"], metric2d=False, gridfile=filename)
gf = xr.open_dataset(filename)
cellvolumes = (gf["J"] * gf["dx"] * gf["dy"] * gf["dz"]).values
cellareas = (
    (np.sqrt(gf["g_11"] * gf["g_33"] - gf["g_13"] * gf["g_13"])) * gf["dx"] * gf["dz"]
)


plotting = False

if plotting:
    testb = polyfunc(np.linspace(0.0, 2.0 * np.pi, 100), field.polynomials)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0.0, 2.0 * np.pi, 100), testb)
    ax.set_ylabel("B [T]")
    ax.set_xlabel(r"$\phi$ [rad]")
    plt.show()

    thisy = 0

    fig, ax = plt.subplots()
    a = ax.pcolormesh(
        gf["R"][:, thisy, :], gf["Z"][:, thisy, :], cellvolumes[:, thisy, :]
    )
    plt.colorbar(a, label="J")

# %% Create the numerical integration

ycoord = np.linspace(0, 2.0 * np.pi, ny, endpoint=False)

thisx = 5
thisy = 3
thisz = 3
n = 500

grid_dl = np.sqrt(gf["g_22"]) * gf["dy"]

thisycoord = np.linspace(
    (ycoord[thisy] + ycoord[thisy - 1]) / 2.0,
    (ycoord[thisy] + ycoord[thisy + 1]) / 2.0,
    n,
)

result = 0.0
arclength = 0.0
for i in range(n - 1):
    thisfield = 0.5 * (
        field.Byfunc(0.0, 0.0, thisycoord[i])
        + field.Byfunc(0.0, 0.0, thisycoord[i + 1])
    )
    dl = (
        (thisycoord[i + 1] - thisycoord[i])
        / (2.0 * np.pi)
        * (gf["R"][thisx, thisy, thisz] * 2.0 * np.pi)
    )
    arclength += dl.values
    result += (
        cellareas[thisx, thisy, thisz]
        * field.Byfunc(0.0, 0.0, ycoord[thisy])
        / thisfield
        * dl
    )


# %% DO the comparison

cellvolumes = {}
error = {}
errors = []
for ref in reffactors:
    cellvolumes[str(ref)] = (
        maps[str(ref)]["J"] * gf["dx"] * gf["dy"] * gf["dz"]
    ).values
    error[str(ref)] = abs(cellvolumes[str(ref)][thisx, thisy, thisz] - result)
    errors.append(error[str(ref)])

coeffs = np.polyfit(np.log(reffactors), np.log(errors), 1)
a, b = coeffs
print("Convergence:", -a)

if plotting:
    fig, ax = plt.subplots()
    for ref in reffactors:
        ax.scatter(ref, error[str(ref)])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.set_xlabel("Refinement factor")
    ax.set_ylabel("Error")
    plt.show()

fail = False

if abs(a) < 1.8:
    fail = True

assert not fail
