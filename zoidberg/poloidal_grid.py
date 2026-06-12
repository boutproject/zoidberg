"""Routines for generating structured meshes on poloidal domains

Classes
-------
`RectangularPoloidalGrid`
    Simple rectangles in R-Z
`StructuredPoloidalGrid`
    Curvilinear structured grids in R-Z

Functions
---------
`grid_annulus`
    Create a `StructuredPoloidalGrid` from inner and outer RZLine
    objects using a simple algorithm
`grid_elliptic`
    Create a `StructuredPoloidalGrid` from inner and outer RZLine
    objects using elliptic meshing method

"""

import warnings

import numpy as np
import shapely
from numpy import linspace, pi, zeros
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree as KDTree

try:
    import matplotlib.pyplot as plt

    plotting_available = True
except ImportError:
    warnings.warn("Couldn't import matplotlib, plotting not available.")
    plotting_available = False

from . import rzline


class PoloidalGrid(object):
    """Represents a poloidal grid

    Note: Here the 2D plane (R,Z) is labelled by (x,z) indices

    Attributes
    ----------
    nx, nz : int
        Number of points in x and z
    R : ndarray
        2D Numpy array of R coordinates
    Z : ndarray
        2D Numpy array of Z coordinates

    """

    def plot(self, axis=None, show=True):
        """Plot grid using matplotlib

        Parameters
        ----------
        axis : matplotlib axis, optional
            A matplotlib axis to plot on. By default a new figure
            is created
        show : bool, optional
            Calls plt.show() at the end

        Returns
        -------
        axis
            The matplotlib axis that was used

        """

        if not plotting_available:
            warnings.warn("matplotlib not available, unable to plot")
            return None

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        axis.plot(self.R, self.Z, "k-")
        axis.plot(self.R.T, self.Z.T, "k-")

        if show:
            plt.show()

        return axis


class RectangularPoloidalGrid(PoloidalGrid):
    """Represents a poloidal grid consisting of a rectangular domain

    Note: Here the 2D plane (R,Z) is labelled by (x,z) indices

    Attributes
    ----------
    nx, nz : int
        Number of points in x and z
    R : ndarray
        2D Numpy array of R coordinates
    Z : ndarray
        2D Numpy array of Z coordinates

    Parameters
    ----------
    nx : int
        Number of points in major radius (including boundaries)
    nz : int
        Number of points in height (including boundaries)
    Lx : float
        Radial domain size  [m]
    Lz : float
        Vertical domain size [m]
    Rcentre : float, optional
        Coordinate at the middle of the domain
    Zcentre : float, optional
        Coordinate at the middle of the domain
    MXG : int, optional
        Number of guard cells in X. The boundary is put half-way
        between the guard cell and the domain

    """

    def __init__(self, nx, nz, Lx, Lz, Rcentre=0.0, Zcentre=0.0, MXG=2):
        self.nx = nx
        self.nz = nz

        self.Lx = Lx
        self.Lz = Lz

        self.Rcentre = Rcentre
        self.Zcentre = Zcentre

        # Some useful derived quantities
        # Note: index at the middle of the domain is (nx - 1)/2
        #       e.g. nx=5 :  0 1 | 2 | 3 4
        self.dR = self.Lx / (self.nx - 2 * MXG)
        self.dZ = self.Lz / self.nz
        self.Rmin = self.Rcentre - self.dR * (self.nx - 1.0) / 2.0
        self.Zmin = self.Zcentre - self.dZ * (self.nz - 1.0) / 2.0

        # Generate 2D arrays
        # Using getCoordinate to ensure consistency
        xind, zind = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        self.R, self.Z = self.getCoordinate(xind, zind)

    def __repr__(self):
        return (
            "RectangularPoloidalGrid({0},{1},{2},{3},Rcentre={4},Zcentre={5})".format(
                self.nx, self.nz, self.Lx, self.Lz, self.Rcentre, self.Zcentre
            )
        )

    def getCoordinate(self, xind, zind, dx=0, dz=0):
        """Get coordinates (R,Z) at given (xind,zind) index

        Parameters
        ----------
        xind, zind : array_like
            Indices in X and Z. These should be the same shape
        dx : int, optional
            Order of x derivative
        dz : int, optional
            Order of z derivative

        Returns
        -------
        R, Z : (ndarray, ndarray)
            Locations of point or derivatives of R,Z with respect to
            indices if dx,dz != 0

        """

        # Convert to NumPy arrays if not already
        xind = np.asarray(xind)
        zind = np.asarray(zind)
        # Make sure dx and dz are integers
        dx = int(dx)
        dz = int(dz)

        assert xind.shape == zind.shape
        assert dx >= 0
        assert dz >= 0

        shape = xind.shape

        if dx + dz > 2:
            # Second derivatives and mixed derivatives all zero
            return np.zeros(shape), np.zeros(shape)

        if dx == 1:
            # dR/dx, dZ/dx
            return np.full(shape, self.dR), np.zeros(shape)
        elif dz == 1:
            # dR/dz, dZ/dz
            return np.zeros(shape), np.full(shape, self.dZ)
        # Return (R,Z) location
        return self.Rmin + xind * self.dR, self.Zmin + zind * self.dZ

    def findIndex(self, R, Z):
        """Finds the (x,z) index corresponding to the given (R,Z) coordinate

        Parameters
        ----------
        R, Z : array_like
            Locations to find indices for

        Returns
        -------
        x, z : (ndarray, ndarray)
            Index as a float, same shape as R,Z

        """

        # Make sure inputs are NumPy arrays
        R = np.asarray(R)
        Z = np.asarray(Z)

        # Check that they have the same shape
        assert R.shape == Z.shape

        xind = (R - self.Rmin) / self.dR
        zind = (Z - self.Zmin) / self.dZ

        # Note: These indices may be outside the domain,
        # but this is handled in BOUT++, and useful for periodic
        # domains.

        return xind, zind

    def metric(self):
        """Return the metric tensor, dx and dz

        For this rectangular grid the metric is the identity

        Returns
        -------
        dict
            Dictionary containing:
            - **dx, dz**: Grid spacing
            - **gxx, gxz, gzz**: Covariant components
            - **g_xx, g_xz, g_zz**: Contravariant components
        """
        return {
            "dx": self.dR,
            "dz": self.dZ,  # Grid spacing
            "gxx": 1.0,
            "g_xx": 1.0,
            "gxz": 0.0,
            "g_xz": 0.0,
            "gzz": 1.0,
            "g_zz": 1.0,
        }


class StructuredPoloidalGrid(PoloidalGrid):
    """Represents a structured poloidal grid in R-Z

    Attributes
    ----------
    nx, nz : int
        Number of points in x and z
    R : ndarray
        2D Numpy array of R coordinates
    Z : ndarray
        2D Numpy array of Z coordinates

    Parameters
    ----------
    R, Z : ndarray
        2D Numpy arrays of R,Z points

        .. note:: R,Z are not copied, so these arrays should not be
                  modified afterwards

    """

    def __init__(self, R, Z):
        assert R.shape == Z.shape

        self.R = R
        self.Z = Z

        nx, nz = R.shape
        self.nx = nx
        self.nz = nz

        if nx > 4:
            inner = shapely.Polygon(zip(R[2, :], Z[2, :]))
            outer = shapely.Polygon(zip(R[-2, :], Z[-2, :]))

            assert (
                inner.area <= outer.area
            ), f"""You are trying to create a grid with inner boundary at high x
and outer boundary at low x. This is against the convention -
switch inner and outer boundary.
                {inner.area} {outer.area}"""

        # Create a KDTree for quick lookup of nearest points
        data = np.concatenate((R.reshape((-1, 1)), Z.reshape((-1, 1))), axis=1)
        self.tree = KDTree(data)

        # Create splines for quick interpolation of coordinates
        xinds = np.arange(nx)
        zinds = np.arange(nz * 3)
        # Repeat the data in z, to approximate periodicity
        R_ext = np.concatenate((R, R, R), axis=1)
        Z_ext = np.concatenate((Z, Z, Z), axis=1)

        self._spl_r = RectBivariateSpline(xinds, zinds, R_ext)
        self._spl_z = RectBivariateSpline(xinds, zinds, Z_ext)

    def __repr__(self):
        return "StructuredPoloidalGrid()"

    def getCoordinate(self, xind, zind, dx=0, dz=0):
        """Get coordinates (R, Z) at given (xind, zind) index

        Parameters
        ----------
        xind, zind : array_like
            Indices in X and Z. These should be the same shape
        dx : int, optional
            Order of x derivative
        dz : int, optional
            Order of z derivative

        Returns
        -------
        R, Z : (ndarray, ndarray)
            Locations of point or derivatives of R,Z with respect to
            indices if dx,dz != 0

        """
        nx, nz = self.R.shape
        if (np.amin(xind) < 0) or (np.amax(xind) > nx - 1):
            raise ValueError("x index out of range")

        # Periodic in y
        zind = np.remainder(zind, nz)

        R = self._spl_r(xind, zind + self.nz, dx=dx, dy=dz, grid=False)
        Z = self._spl_z(xind, zind + self.nz, dx=dx, dy=dz, grid=False)

        return R, Z

    def findIndex(self, R, Z, tol=1e-12, show=False):
        """Finds the (x, z) index corresponding to the given (R, Z) coordinate

        Parameters
        ----------
        R, Z : array_like
            Locations. Can be scalar or array, must be the same shape
        tol : float, optional
            Maximum tolerance on the square distance

        Returns
        -------
        x, z : (ndarray, ndarray)
            Index as a float, same shape as R, Z

        """

        # Make sure inputs are NumPy arrays
        R = np.asarray(R)
        Z = np.asarray(Z)

        # Check that they have the same shape
        assert R.shape == Z.shape

        input_shape = R.shape  # So output has same shape as input

        # Get distance and index into flattened data
        # Note ind can be an integer, or an array of ints
        # with the same number of elements as the input (R,Z) arrays
        n = R.size
        position = np.concatenate((R.reshape((n, 1)), Z.reshape((n, 1))), axis=1)

        R = R.reshape((n,))
        Z = Z.reshape((n,))

        dists, ind = self.tree.query(position)

        # Calculate (x,y) index
        nx, nz = self.R.shape
        xind = np.floor_divide(ind, nz)
        zind = ind - xind * nz

        # Convert indices to float
        xind = np.asarray(xind, dtype=float)
        zind = np.asarray(zind, dtype=float)

        # Create a mask for the positions
        mask = np.ones(xind.shape)
        mask[np.logical_or((xind < 0.5), (xind > (nx - 1.5)))] = (
            0.0  # Set to zero if near the boundary
        )

        if show and plotting_available:
            plt.plot(self.R, self.Z, ".")
            plt.plot(R, Z, "x")

        cnt = 0
        underrelax = 1
        extra = 0

        while True:
            # Use Newton iteration to find the index
            # dR, dZ are the distance away from the desired point
            Rpos, Zpos = self.getCoordinate(xind, zind)
            if show and plotting_available:
                plt.plot(Rpos, Zpos, "o")
            dR = Rpos - R
            dZ = Zpos - Z

            # Check if close enough
            # Note: only check the points which are not in the boundary
            val = np.amax(mask * (dR**2 + dZ**2))
            if val < tol:
                extra += 1
                if extra > 3:
                    break
            else:
                if extra:
                    raise RuntimeError("Failed to converge")
            cnt += 1
            if cnt == 10:
                underrelax = 1.5
            if cnt == 100:
                underrelax = 2
            if cnt == 300:
                underrelax = 2.5
            if cnt == 700:
                underrelax = 3
            if cnt == 1000:
                raise RuntimeError("Failed to converge")

            # Calculate derivatives
            dRdx, dZdx = self.getCoordinate(xind, zind, dx=1)
            dRdz, dZdz = self.getCoordinate(xind, zind, dz=1)

            # Invert 2x2 matrix to get change in coordinates
            #
            # (x) -=  ( dR/dx   dR/dz )^-1  (dR)
            # (y)     ( dZ/dx   dZ/dz )     (dz)
            #
            #
            # (x) -=  ( dZ/dz  -dR/dz ) (dR)
            # (y)     (-dZ/dx   dR/dx ) (dZ) / (dR/dx*dZ/dy - dR/dy*dZ/dx)
            determinant = dRdx * dZdz - dRdz * dZdx

            xind -= mask * ((dZdz * dR - dRdz * dZ) / determinant / underrelax)
            zind -= mask * ((dRdx * dZ - dZdx * dR) / determinant / underrelax)

            # Re-check for boundary
            in_boundary = xind < 0.5
            mask[in_boundary] = 0.0  # Set to zero if near the boundary
            xind[in_boundary] = 0.0
            out_boundary = xind > (nx - 1.5)
            mask[out_boundary] = 0.0  # Set to zero if near the boundary
            xind[out_boundary] = nx - 1

        if show and plotting_available:
            plt.show()

        # Set xind to -1 if in the inner boundary, nx if in outer boundary
        in_boundary = xind < 0.5
        xind[in_boundary] = -1
        out_boundary = xind > (nx - 1.5)
        xind[out_boundary] = nx

        return xind.reshape(input_shape), zind.reshape(input_shape)

    def metric(self):
        """Return the metric tensor, dx and dz

        Returns
        -------
        dict
            Dictionary containing:
            - **dx, dz**: Grid spacing
            - **gxx, gxz, gzz**: Covariant components
            - **g_xx, g_xz, g_zz**: Contravariant components

        """

        # Get arrays of indices
        xind, zind = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing="ij")

        # Calculate the gradient along each coordinate
        dolddnew = np.array(
            [self.getCoordinate(xind, zind, dx=a, dz=b) for a, b in ((1, 0), (0, 1))]
        )
        # dims: 0 : dx or dz?
        #       1 : R or z?
        #       2 : spatial: r
        #       3 : spatial: \theta
        ddist = np.sqrt(np.sum(dolddnew**2, axis=1))  # sum R + z
        nx, nz = ddist.shape[1:]
        ddist[0] = 1 / nx
        ddist[1] = 1 / nz
        dolddnew /= ddist[:, None, ...]

        # g_ij = J_ki J_kj
        # (2.5.27) from D'Haeseleer 1991
        # Note: our J is transposed
        J = dolddnew
        g = np.sum(
            np.array(
                [
                    [[J[j, i] * J[k, i] for i in range(2)] for j in range(2)]
                    for k in range(2)
                ]
            ),
            axis=2,
        )

        assert np.all(g[0, 0] > 0), (
            f"g[0, 0] is expected to be positive, but some values are not (minimum {np.min(g[0, 0])})"
        )
        assert np.all(g[1, 1] > 0), (
            f"g[1, 1] is expected to be positive, but some values are not (minimum {np.min(g[1, 1])})"
        )
        g = g.transpose(2, 3, 0, 1)
        assert np.all(np.linalg.det(g) > 0), (
            f"All determinants of g should be positive, but some are not (minimum {np.min(np.linalg.det(g))})"
        )
        ginv = np.linalg.inv(g)
        return {
            "dx": ddist[0],
            "dz": ddist[1],  # Grid spacing
            "gxx": ginv[..., 0, 0],
            "g_xx": g[..., 0, 0],
            "gxz": ginv[..., 0, 1],
            "g_xz": g[..., 0, 1],
            "gzz": ginv[..., 1, 1],
            "g_zz": g[..., 1, 1],
        }


def grid_annulus(inner, outer, nx, nz, show=True, return_coords=False):
    """Grid an annular region, given inner and outer boundaries both of
    which are RZline objects

    This is a very simple algorithm which just draws straight lines
    between inner and outer boundaries.

    Parameters
    ----------
    inner, outer : `RZline`
        Inner and outer boundaries of the domain
    nx : int
        The required radial resolution, including boundaries
    nz : int
        The required poloidal resolution
    show : bool, optional
        If True, plot the resulting grid
    return_coords : bool, optional
        If True, return the R, Z coordinates of the grid points,
        instead of a `StructuredPoloidalGrid`

    Returns
    -------
    StructuredPoloidalGrid
        A grid of the region

    """

    assert nx >= 2
    assert nz > 1

    R = zeros((nx, nz))
    Z = zeros((nx, nz))

    # Generate angle values, which should now be equally spaced
    # in distance along inner and outer boundaries
    thetavals = linspace(0, 2 * pi, nz, endpoint=False)

    # Radial coordinate
    xvals = linspace(0, 1.0, nx, endpoint=True)

    innerR = inner.Rvalue(thetavals)
    innerZ = inner.Zvalue(thetavals)

    outerR = outer.Rvalue(thetavals)
    outerZ = outer.Zvalue(thetavals)
    for i, x in enumerate(xvals):
        # Get the R and Z coordinates of this line
        R[i, :] = x * outerR + (1.0 - x) * innerR
        Z[i, :] = x * outerZ + (1.0 - x) * innerZ

    if show and plotting_available:
        plt.plot(inner.R, inner.Z, "-o")
        plt.plot(outer.R, outer.Z, "-o")

        plt.plot(R, Z, "x")

        plt.show()

    if return_coords:
        return R, Z
    return StructuredPoloidalGrid(R, Z)


class PlotDelayed:
    def __init__(self):
        self.store = {}

    def add(self, R, Z, label):
        self.store[label] = R.copy(), Z.copy()

    def plot(self, block=True):
        for k, v in self.store.items():
            plt.figure()
            R, Z = v
            plt.plot(R, Z)
            plt.plot(R.T, Z.T)
            plt.title(k)

        plt.show(block=block)

    def clear(self):
        self.store = {}


def contains(inner, outer, pnt):
    import eudist

    if eudist.winding_number(inner, pnt) == 0:
        return eudist.winding_number(outer, pnt) != 0
    return False


class ForceField:
    def __init__(self, mins, maxs, step, dims=1):
        # import itertools
        self.mins = mins
        self.step = step
        dists = (maxs - mins) / step
        dists = np.ceil(dists).astype(int)
        self.field = np.zeros((*dists, dims))

    def getAt(self, point):
        offsets = (point - self.mins) / self.step
        if any(offsets < 0) or any(offsets - 1 > self.field.shape[:-1]):
            return np.zeros(self.field[-1])
        offset_ints = np.floor(offsets).astype(int)
        facs = offsets - offset_ints
        slc = [slice(i, i + 2) for i in offset_ints]
        local = self.field[slc]
        for f in facs:
            assert 0 <= f < 1, facs
            local = local[0] * (1 - f) + local[1] * f
        return local


class ForceFieldPoor:
    def __init__(self, mins, maxs, step, dims=1):
        # import itertools
        self.mins = mins
        self.step = step
        dists = (maxs - mins) / step
        dists = np.ceil(dists).astype(int)
        self.field = np.zeros((*dists, dims))

    def getAt(self, point):
        offsets = np.round((point - self.mins) / self.step).astype(int)
        if any(offsets < 0):
            return np.zeros(self.field[-1])
        try:
            return self.field[*offsets]
        except IndexError:
            return np.zeros(self.field[-1])


def calc_force_field(field, RZi):
    F = np.zeros_like(RZi)
    for j in range(RZi.shape[1]):
        F[:, j] = field.getAt(RZi[:, j])
    return F


def calc_force_part(useinner, inner, outer, RZi):
    import eudist
    import numpy as np

    F = np.zeros_like(RZi)
    for j in range(RZi.shape[1]):
        oij = slice(None), j
        #
        # inside = domain.contains(point)
        inside = contains(inner, outer, RZi[oij])
        dist = -eudist.line_segments_dot_vec(
            inner if useinner else outer, RZi[oij], periodic=True
        )

        dist1d = np.sqrt(np.sum(dist**2))
        sign = -1 if inside else +1
        maxdist = 0.005
        falloff = 5e2
        if (
            dist1d > 1e-8
            and (dist1d < maxdist and inside)
            or (not inside and dist1d < 0.05)
        ):
            force = (
                sign
                * dist
                / dist1d
                * (np.exp(dist1d * sign * falloff) - np.exp(-maxdist * falloff))
            )
            F[:, j] -= force / 1e3
    return F


class BucketMesh:
    def __init__(self, polygon, maxdist):
        from tqdm.contrib import itertools
        import eudist

        self.md = maxdist
        mins = np.min(polygon, axis=0)
        maxs = np.max(polygon, axis=0)
        print("bucket", mins, maxs, polygon.shape)
        mins = np.floor(mins / maxdist) - 1
        maxs = np.ceil(maxs / maxdist) + 1
        dists = maxs - mins
        print(maxs, mins, dists)
        dists = dists.astype(int)
        self.mins = mins
        self.maxs = maxs
        self.buckets = np.zeros(dists, dtype=object)
        for ij in itertools.product(*[range(x) for x in dists]):
            coords = (np.array(ij) - self.mins) * self.md
            for offsets in itertools.product(*[(0, maxdist) for _ in mins]):
                for i in range(polygon.shape[1]):
                    if eudist.line_segment_dot(polygon[i - 1], polygon):
                        pass
                coords + np.array(offsets)
        print(dists)
        assert False


class _allGood:
    def __init__(self, R, Z):
        self.outer = shapely.LinearRing(zip(R[0], Z[0]))
        self.inner = shapely.LinearRing(zip(R[-1], Z[-1]))

    def check(self, R, Z, onlyIO=False):
        i = -1
        last = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
        for i in range(R.shape[1]):
            cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
            if cur.intersects(self.inner) or cur.intersects(self.outer):
                return False
            if onlyIO:
                if cur.intersects(last):
                    return False
            last = cur
        return True

    def checkOuter(self, R, Z):
        for i in range(R.shape[1]):
            cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
            if cur.intersects(self.outer):
                return False
        return True


def _ensure_monotonic_force(R, Z):
    import eudist
    from tqdm.contrib import itertools

    RZ = np.array((R, Z))
    nx, nz = R.shape
    outer = shapely.LineString(RZ[:, -1, :].T)
    inner = shapely.LineString(RZ[:, 0, :].T)
    # domain = shapely.Polygon(outer, [inner])
    inner_eudist = np.ascontiguousarray(RZ[:, 0, :].T)
    outer_eudist = np.ascontiguousarray(RZ[:, -1, :].T)

    # inner_buckets = BucketMesh(inner_eudist, 0.05)
    if 0:
        mins = np.min(outer_eudist, axis=0) - 0.07
        if 0:
            step = 0.01
            force_field = ForceField(mins, np.max(outer_eudist, axis=0) + 0.07, step, 2)
        else:
            step = 0.001
            force_field = ForceFieldPoor(
                mins, np.max(outer_eudist, axis=0) + 0.07, step, 2
            )
        for ij in itertools.product(*[range(x) for x in force_field.field.shape[:2]]):
            aij = np.array(ij) * step
            pos = aij + mins
            inside = contains(inner_eudist, outer_eudist, pos)
            for poly in (inner_eudist, outer_eudist):
                dist = -eudist.line_segments_dot_vec(poly, pos, periodic=True)

                dist1d = np.sqrt(np.sum(dist**2))
                sign = -1 if inside else +1
                maxdist = 0.005
                falloff = 5e3
                if (
                    dist1d > 1e-8
                    and (dist1d < maxdist and inside)
                    or (not inside and dist1d < 0.05)
                ):
                    force = (
                        sign
                        * dist
                        / dist1d
                        * (np.exp(dist1d * sign * falloff) - np.exp(-maxdist * falloff))
                    )
                    force_field.field[ij] -= force / 1e3
        if 0:
            plt.figure()
            Rf = np.arange(0, force_field.field.shape[0]) * step + mins[0]
            Zf = np.arange(0, force_field.field.shape[1]) * step + mins[1]
            plt.quiver(
                Rf[:, None] + 0 * Zf[None, :],
                0 * Rf[:, None] + Zf[None, :],
                force_field.field[:, :, 0],
                force_field.field[:, :, 1],
            )
            plt.plot(*RZ[:, 0], "r-")
            plt.plot(*RZ[:, -1], "r-")
            plt.gca().set_aspect(1)
            plt.figure()
            plt.pcolormesh(
                Rf[:, None] + 0 * Zf[None, :],
                0 * Rf[:, None] + Zf[None, :],
                force_field.field[:, :, 0],
            )
            plt.colorbar()
            plt.plot(*RZ[:, 0], "r-")
            plt.plot(*RZ[:, -1], "r-")
            plt.gca().set_aspect(1)
            plt.figure()
            plt.pcolormesh(
                Rf[:, None] + 0 * Zf[None, :],
                0 * Rf[:, None] + Zf[None, :],
                force_field.field[:, :, 1],
                vmax=10,
            )
            plt.colorbar()
            plt.plot(*RZ[:, 0], "r-")
            plt.plot(*RZ[:, -1], "r-")
            plt.gca().set_aspect(1)
            plt.show(block=True)

    assert inner_eudist.flags["C_CONTIGUOUS"]

    def allGood(R, Z):
        i = -1
        last = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
        for i in range(R.shape[1]):
            cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
            if cur.intersects(inner) or cur.intersects(outer):
                return False
            if cur.intersects(last):
                return False
            last = cur
        return True

    def contains_eudist(pnt):
        if eudist.winding_number(inner_eudist, pnt) == 0:
            return eudist.winding_number(outer_eudist, pnt) != 0
        return False

    def contains_(pnt):
        tmp = contains_eudist(pnt)
        return tmp

    def force(RZ, scale=1):
        Dx = 10
        Dz = 1
        RZi = RZ[:, 1:-1]
        Fz = 0
        Fx = 0
        Fz = np.roll(RZi, 1, axis=-1) - RZi
        Fz += np.roll(RZi, -1, axis=-1) - RZi
        Fx = RZ[:, 2:] - RZi
        Fx += RZ[:, :-2] - RZi
        F = Fz * Dz + Fx * Dx
        # from concurrent.futures import InterpreterPoolExecutor as Executor
        # from concurrent.futures import ThreadPoolExecutor as Executor
        from concurrent.futures import ProcessPoolExecutor as Executor

        if False:
            with Executor(max_workers=64) as executor:
                futures = [
                    executor.submit(
                        calc_force_part,
                        i < nx // 2,
                        inner_eudist,
                        outer_eudist,
                        RZ[:, i],
                    )
                    for i in range(1, RZ.shape[1] - 1)
                ]
                for i, f in enumerate(futures):
                    F[:, i] += f.result()
        elif 0:
            for i in range(1, RZ.shape[1] - 1):
                Fi = calc_force_field(force_field, RZ[:, i])
                F[:, i - 1] += Fi
        else:
            for i in range(1, RZ.shape[1] - 1):
                Fi = calc_force_part(i < nx // 2, inner_eudist, outer_eudist, RZ[:, i])
                F[:, i - 1] += Fi
        return F, Fz * Dz, Fx * Dx

    def force_1d(t, y):
        RZ[:, 1:-1] = y.reshape((2, nx - 2, nz))
        F = force(RZ)
        return F.ravel()

    def jacobian(t, y):
        pass

    def sparsity(nx, nz):
        N = nx * nz
        # ret = scipy.sparse.bsr_array((N, N), dtype=bool)
        ret = np.zeros((N, N), dtype=bool)

        def ind(i, j):
            return i * nz + (j % nz)

        for i in range(nx):
            for j in range(nz):
                ij = ind(i, j)
                ret[ij, ij] = True
                ret[ij, ind(i, j + 1)] = True
                ret[ij, ind(i, j - 1)] = True
                if i > 0:
                    ret[ij, ind(i - 1, j)] = True
                if i < nx - 1:
                    ret[ij, ind(i + 1, j)] = True

    scale = 1
    todo = 1
    noPlot = 1000
    import sys
    import select
    import tty
    import termios

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def getData():
        if isData():
            return sys.stdin.read(1)
        return None

    stepsize = 1e-2
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while todo != 0:
            # not allGood(*RZ):
            if allGood(*RZ):
                todo -= 1
            else:
                todo += 1
            scale *= 0.998
            try:
                F, Fz, Fx = force(RZ, scale)
            except:
                plt.figure()
                plt.quiver(*RZ[:, 1:-1], *F)
                plt.plot(*RZ[:, 0], "r-")
                plt.plot(*RZ[:, -1], "r-")
                plt.figure()
                plt.plot(*RZ)
                plt.plot(*RZ.transpose(0, 2, 1))
                plt.figure()
                for ind in 0, 1, -2, -1:
                    plt.plot(*RZ[:, ind], label=ind)
                plt.legend()
                plt.show(block=True)
                raise

            maxforce = np.max(np.sqrt(np.sum(F**2, axis=0)))
            print(noPlot, todo, scale, maxforce, RZ.shape, stepsize)

            d = getData()
            if d:
                if d == "p":
                    print("Plotting forces!")
                    noPlot = 1000
                    RZm = RZ[:, 1:] + RZ[:, :-1]
                    RZm /= 2
                    plt.figure()
                    plt.quiver(*RZ[:, 1:-1], *F, color="black")
                    plt.quiver(*RZ[:, 1:-1], *Fx, color="green")
                    plt.quiver(*RZ[:, 1:-1], *Fz, color="pink")
                    plt.quiver(*RZ[:, 1:-1], *(F - Fz - Fx), color="blue")
                    plt.plot(*RZ[:, 0], "r-")
                    plt.plot(*RZ[:, -1], "r-")
                    plt.gca().set_aspect(1)
                    plt.show(block=True)
                if d == "g":
                    print("Plotting grid")
                    plt.figure()
                    plt.plot(*RZ.transpose(0, 2, 1))
                    plt.plot(*RZ)
                    plt.gca().set_aspect(1)
                    plt.show(block=True)
                if d == "e":
                    todo = 0
                if d == "r":
                    stepsize /= 2
                if d == "i":
                    stepsize *= 2

            RZ[:, 1:-1] += F / maxforce * stepsize
            noPlot -= 1
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return RZ


def _ensure_monotonic(R, Z, overshoot=1.05, do=None, args=[]):
    import shapely
    import shapely.ops

    outer = shapely.LinearRing(zip(R[-1, :], Z[-1, :]))
    inner = shapely.LinearRing(zip(R[0, :], Z[0, :]))

    def push_in(R, Z, overshoot, overshoot2=None, restore=0.5):
        counter = 0
        plotter = PlotDelayed()
        plotter.add(R, Z, "before")
        # push points inside
        for i in range(R.shape[0] - 2, 0, -1):
            area = shapely.Polygon(zip(R[i + 1, :], Z[i + 1, :]))
            for j in range(R.shape[1]):
                pnt = shapely.Point(R[i, j], Z[i, j])
                if not area.contains(pnt):
                    p1, p2 = shapely.ops.nearest_points(area, pnt)
                    dx = p1.x - p2.x
                    dz = p1.y - p2.y
                    R[i, j] += dx * overshoot
                    Z[i, j] += dz * overshoot

        # push lines inside
        for i in range(R.shape[0] - 1, 1, -1):
            if overshoot2 is not None:
                overshoot = overshoot2
            line = shapely.LinearRing(zip(R[i - 1, :], Z[i - 1, :]))
            area = shapely.Polygon(line, [inner])
            for j in range(R.shape[1]):
                pnt = shapely.Point(R[i, j], Z[i, j])
                if area.contains(pnt):
                    doplotnow = False
                    if doplotnow:
                        plt.plot(R[i - 1], Z[i - 1], ls="--", color="grey")
                    p1, p2 = shapely.ops.nearest_points(line, pnt)
                    dx = p1.x - p2.x
                    dz = p1.y - p2.y
                    # proposed = R[i-1, j], Z[i-1, j]
                    # for k, dd in enumerate((dx, dz)):
                    #     proposed[k] -= dd * overshoot
                    # for k, org in enumerate((R, Z)):
                    #     proposed[k] = (proposed[k] + org[i, j]) / 2

                    R[i - 1, j] -= dx * overshoot
                    Z[i - 1, j] -= dz * overshoot
                    R[i - 1, j] = R[i - 1, j] * restore + R[i, j] * (1 - restore)
                    Z[i - 1, j] = Z[i - 1, j] * restore + Z[i, j] * (1 - restore)

                    if doplotnow:
                        plt.plot(R[i - 1 : i + 1], Z[i - 1 : i + 1])
                        plt.plot(R[i - 1 : i + 1].T, Z[i - 1 : i + 1].T)
                        for pi in p1, p2:
                            plt.plot(pi.x, pi.y, "ro")
                        plt.gca().set_aspect(1)
                        plt.show(block=True)
                    line = shapely.LinearRing(zip(R[i - 1, :], Z[i - 1, :]))
                    area = shapely.Polygon(line, [inner])

        counter += 1
        overshoot = (overshoot + 1) / 2
        plotter.add(R, Z, f"iteration {counter}")
        if counter > 3:
            plotter.plot()
            raise RuntimeError("Failed to push points in/out")
        return R, Z

    def push_poloidally(R, Z, overshoot, doplot=False, other=False):
        i = -1
        isgood = np.empty(R.shape[1], dtype=bool)
        last = shapely.LineString(zip(R[:, i], Z[:, i]))
        for i in range(0, R.shape[1]):
            cur = shapely.LineString(zip(R[:, i], Z[:, i]))
            if cur.intersects(last):
                outerr = R[-1], Z[-1]
                inerr = R[0], Z[0]
                curisgood = shapely.Point(R[-1, i], Z[-1, i]).distance(
                    last
                ) < shapely.Point(R[-1, i - 1], Z[-1, i - 1]).distance(cur)
                if curisgood:
                    isgood[i] = False
                else:
                    isgood[i - 1] = False
            last = cur

        last = shapely.LineString(zip(R[:, i], Z[:, i]))
        frac = R.shape[1] // 2
        plotter = PlotDelayed()
        plotter.add(R, Z, "before")
        for i in range(0, R.shape[1]):
            cur = shapely.LineString(zip(R[:, i], Z[:, i]))
            if cur.intersects(last):
                outerr = R[-1], Z[-1]
                inerr = R[0], Z[0]
                curisgood = shapely.Point(R[-1, i], Z[-1, i]).distance(
                    last
                ) < shapely.Point(R[-1, i - 1], Z[-1, i - 1]).distance(cur)
                if other:
                    curisgood = not curisgood
                if curisgood:
                    # last is bad
                    badi = i - 1
                    goodi = i
                    slc = slice(-frac, -1)
                    direc = -1
                else:
                    badi = i
                    goodi = i - 1
                    slc = slice(0, frac)
                    direc = 1
                outerr = [np.roll(x, -goodi + direc) for x in outerr]
                outerr = [x[slc][::direc] for x in outerr]
                inerr = [np.roll(x, -goodi + direc) for x in inerr]
                inerr = [x[slc][::-direc] for x in inerr]
                # print(slc, outerr[0].shape)
                # plt.figure()
                allowed = [np.array([]), np.array([])]
                for a, b in (
                    (R[:, goodi], Z[:, goodi]),
                    outerr,
                    (R[::-1, goodi - frac], Z[::-1, goodi - frac]),
                    inerr,
                ):
                    allowed[0] = np.append(allowed[0], a)
                    allowed[1] = np.append(allowed[1], b)
                # print("ind", goodi, goodi-frac)
                # plt.plot(*allowed, "g-")
                # plt.plot(R[1:-1, badi], Z[1:-1, badi], "rx")
                # print(R[:, goodi], Z[:, goodi], "b-")
                # print(R[:, goodi-frac], Z[:, goodi-frac], "o-")
                # plt.title("HERE")
                # plt.show(block=True)
                poly = shapely.Polygon(shapely.LineString(zip(*allowed)))
                for j in range(1, R.shape[0] - 1):
                    pnt = shapely.Point(R[j, badi], Z[j, badi])
                    if not poly.contains(pnt):
                        p1, p2 = shapely.ops.nearest_points(poly, pnt)
                        dx = p1.x - p2.x
                        dz = p1.y - p2.y
                        R[j, badi] += dx * overshoot
                        Z[j, badi] += dz * overshoot

            cur = shapely.LineString(zip(R[:, i], Z[:, i]))
            if cur.intersects(last) and doplot:
                plotter.add(R, Z, "after")
                plotter.plot()

            last = cur
        return R, Z

    def push_poloidally_all(R, Z, fac=0.5):
        i = -1
        isgood = np.ones(R.shape[1], dtype=bool)
        last = shapely.LineString(zip(R[:, i], Z[:, i]))
        for i in range(0, R.shape[1]):
            cur = shapely.LineString(zip(R[:, i], Z[:, i]))
            if cur.intersects(last):
                curisgood = shapely.Point(R[-1, i], Z[-1, i]).distance(
                    last
                ) < shapely.Point(R[-1, i - 1], Z[-1, i - 1]).distance(cur)
                if not curisgood:
                    isgood[i] = False
                else:
                    isgood[i - 1] = False
            last = cur
        print(isgood)
        assert isgood[-1]
        for i in range(len(isgood)):
            if not isgood[i]:
                cur = shapely.LineString(zip(R[:, i], Z[:, i]))
                for igood in i - 1, i + 1:
                    other = shapely.LineString(zip(R[:, igood], Z[:, igood]))
                    if not cur.intersects(other):
                        print(f"Moving {i} in direction {igood}")
                        R[1:-1, i] = R[1:-1, i] * fac + R[1:-1, igood] * (1 - fac)
        return R, Z

    def allGood():
        def doplot():
            plt.figure()
            plt.plot(R, Z)
            plt.plot(R.T, Z.T)
            plt.show(block=True)
            assert False

        i = -1
        last = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
        for i in range(R.shape[1]):
            cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
            if cur.intersects(inner) or cur.intersects(outer):
                print("leave domain", i)
                doplot()
                return False
            if cur.intersects(last):
                print("overlap last", i)
                doplot()
                return False
            last = cur
        return True

    def plotBad(R, Z):
        plt.figure()
        for ind in -1, 0:
            plt.plot(R[ind, :], Z[ind, :])
        i = -1
        last = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
        for i in range(R.shape[1]):
            cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
            if cur.intersects(inner) or cur.intersects(outer):
                plt.plot(R[:, i], Z[:, i], "x-")
                plt.plot(R[1:-1, i], Z[1:-1, i], "x-")
            if cur.intersects(last):
                plt.plot(R[1:-1, i], Z[1:-1, i], "x-", color="red")
                plt.plot(R[1:-1, i - 1], Z[1:-1, i - 1], "x-", color="magenta")
            last = cur
            # for bound in inner, outer:
            #     if cur.intersects(bound):
            #         inter = cur.intersection(bound)
            #         print(inter)
            #         plt.plot(inter.x, inter.y, "ro")
        plt.show(block=True)

    if do is not None:
        todos = dict(pol=push_poloidally, rad=push_in, pol2=push_poloidally_all)
        return todos[do](R, Z, *args)
    p = PlotDelayed()
    p.add(R, Z, "init")
    if allGood():
        return R, Z
    R, Z = push_poloidally(R, Z, overshoot)
    p.add(R, Z, "pol1")
    for a, b in ((1.05, 0.5), (1.1, 0.3)):
        if allGood():
            return R, Z
        R, Z = push_in(R, Z, overshoot, 1.5, restore=0.3)
    p.add(R, Z, "in1")
    return R, Z
    if allGood():
        return R, Z
    R, Z = push_poloidally(R, Z, overshoot)
    p.add(R, Z, "pol2")
    if allGood():
        return R, Z
    R, Z = push_poloidally(R, Z, overshoot, other=True)
    if allGood():
        return R, Z
    R, Z = push_poloidally_all(R, Z, 0.5)
    for a, b in ((overshoot, 0.5), (1.5, 0.3), (1.8, 0.2), (2.2, 0.2)):
        if allGood():
            return R, Z
        R, Z = push_in(R, Z, overshoot, a, restore=b)
        if allGood():
            return R, Z
        R, Z = push_poloidally(R, Z, overshoot)
        R, Z = push_poloidally(R, Z, 1.5)

        if allGood():
            return R, Z
    R, Z = push_poloidally_all(R, Z, 0.5)
    if allGood():
        return R, Z
    p.plot(block=False)
    plotBad(R, Z)
    # raise RuntimeError("Failed to make it monotonic")
    # for i in range( R.shape[0] -2, 0, -1):
    #     outer = shapely.LineString(zip(R[i+1, :], Z[i+1, :]))
    #  #    poly = shapely.Polygon(outer)
    # #     for j in range(R.shape[1]):
    # #         pnt = shapely.Point(R[i, j], Z[i, j])
    # #         if not poly.contains(pnt):
    # #             p1, p2 = shapely.ops.nearest_points(poly, pnt)
    # #             dx = p1.x - p2.x
    # #             dz = p1.y - p2.y
    # #             R[i, j] += dx * 1.1
    # #             Z[i, j] += dz * 1.1

    # i = -1
    # last = shapely.LineString(zip(R[:, i], Z[:, i]))
    # for i in range(1, R.shape[1]):
    #     cur = shapely.LineString(zip(R[:, i], Z[:, i]))
    #     if cur.intersects(last):
    #         if shapely.Point(R[-1, i], Z[-1, i]).distance(last) <  shapely.Point(R[-1, i-1], Z[-1, i-1]).distance(cur):
    #             # last is bad
    #             lastl='r-'
    #             curl='g-'
    #         else:
    #             lastl='g-'
    #             curl='r-'
    #         print(i, i-1)
    #         plt.figure()
    #         plt.plot(R[:, i], Z[:, i], curl)
    #         plt.plot(R[:, i-1], Z[:, i-1], lastl)
    #         # inter = cur.intersection(last)
    #         # if isinstance(inter, shapely.geometry.point.Point):
    #         #     print(R.shape, i, inter.xy)
    #         # elif isinstance(inter, shapely.geometry.multipoint.MultiPoint):
    #         #     print(R.shape, i, len(inter.geoms))
    #         #     # print(R.shape, i, inter.coords)
    #         # else:
    #         #     print(i, type(inter))

    #         # plt.figure()
    #         # for j in i, i - 1:
    #         #     plt.plot(R[:, j], Z[:, j])
    #         # for j in -1, 0:
    #         #     plt.plot(R[j, :], Z[j, :])
    #         # if isinstance(inter, shapely.geometry.point.Point):
    #         #     plt.plot(inter.x, inter.y, "ro")
    #         # elif isinstance(inter, shapely.geometry.multipoint.MultiPoint):
    #         #     for pnt in inter.geoms:
    #         #         plt.plot(pnt.x, pnt.y, "ro")

    # #
    # last = cur
    # if 1:
    #     plt.figure()
    #     plt.plot(R.T, Z.T)
    #     plt.plot(R, Z)
    # isbad = np.zeros((R.shape[1]), dtype=bool)
    # for i in range(R.shape[1]):
    #     cur = shapely.LineString(zip(R[1:-1, i], Z[1:-1, i]))
    #     if cur.intersects(inner) or cur.intersects(outer):
    #         isbad[i] = True
    # print(np.sum(isbad))

    # badchunks = []
    # i = -1
    # last = isbad[i]
    # if last:
    #     while isbad[i]:
    #         i -= 1
    #     badchunks.append([i + 1])
    # for i in range(R.shape[1]):
    #     if isbad[i] and not last:
    #         badchunks.append([i])
    #     elif not isbad[i] and last:
    #         badchunks[-1].append(i - 1)
    #     # print(i, isbad[i], last, badchunks)
    #     last = isbad[i]
    # if len(badchunks[-1]):
    #     badchunks = badchunks[:-1]
    # print(badchunks)

    # for zs, ze in badchunks:
    #     prev = R[:, zs - 1], Z[:, zs - 1]
    #     next = R[:, ze + 1], Z[:, ze + 1]
    #     dz = ze - zs + 3
    #     print(zs, ze, dz)
    #     assert dz > 0
    #     fac = np.linspace(0, 1, dz)[None, 1:-1]
    #     fill = [b[:, None] * fac + a[:, None] * (1 - fac) for a, b in zip(prev, next)]
    #     zep = None if ze == -1 else ze + 1
    #     print(fill[0].shape, zs, zep)
    #     R[:, zs:zep] = fill[0]
    #     Z[:, zs:zep] = fill[1]

    # # assert allgood
    return R, Z


def _find_inner(outer, inner):
    rz = rzline.RZline(*inner.T)
    thetavals_inner = [rz.closestPoint(*x) for x in outer.T]
    thetavals_inner = _laplace(thetavals_inner, 50, minsteps=0)
    return np.array((rz.Rvalue(thetavals_inner), rz.Zvalue(thetavals_inner)))


def _get_guess(target_outer, target_inner, nx, nz):
    import shapely.geometry as sg

    outer_ring = sg.LinearRing(target_outer.T)
    inner_ring = sg.LinearRing(target_inner.T)
    inter = _get_intermediate_contours(inner_ring, outer_ring, nx)
    grid_guess = np.empty((2, nx, nz))
    last = target_outer
    grid_guess[:, 0, :] = last

    for k, i in enumerate(inter):
        grid_guess[:, k + 1, :] = _find_inner(last, i)
    grid_guess[:, -1, :] = target_inner  # _find_inner(target_outer.T, target_inner)
    return grid_guess


def _get_intermediate_contours(inner_poly, outer_poly, n_levels):
    max_buffer = inner_poly.hausdorff_distance(outer_poly)

    def buffered_inner(d):
        return inner_poly.buffer(d, join_style="round", cap_style="round")

    # Find max buffer distance where buffered inner fits inside outer
    lo, hi = 0.0, max_buffer * 2
    for _ in range(10):
        mid = (lo + hi) / 2
        buffered = buffered_inner(mid)
        if outer_poly.distance(buffered):
            lo = mid
        else:
            hi = mid
    max_valid_buffer = lo
    print(
        f"Max valid buffer distance: {max_valid_buffer:.10f}; started with {max_buffer:.10f}"
    )
    out = []
    for x in np.linspace(0, max_valid_buffer, n_levels)[1:-1]:
        buf = outer_poly.buffer(x)
        assert len(buf.interiors) == 1, len(buf.interiors)
        out.append(np.array(buf.interiors[0].coords))
    return out


def _laplace(x, maxfac_inner, minsteps=100):

    def dx(x, sign):
        dxv = np.roll(x, sign) - x
        dxv = np.remainder(dxv + np.pi, 2 * np.pi) - np.pi
        return dxv

    nz = len(x)

    fac = 0.01 * nz / 192
    steps = minsteps
    if fac > 0.01:
        steps = int(steps * fac / 0.1)
        fac = 0.01
    x0 = x.copy()
    for i in range(steps):
        x += fac * (dx(x, 1) + dx(x, -1))

    extra = 0
    if maxfac_inner:
        assert maxfac_inner >= 1, "maxfac_inner needs to be at least 1"

        def getfac(x):
            d = dx(x, -1)
            return np.max(d) / np.min(d)

        while not np.all(dx(x, -1) > 0):
            x += fac * (dx(x, 1) + dx(x, -1))
            extra += 1

        if maxfac_inner == 1:
            x1 = x.copy()
            for i in range(1, len(x0)):
                if x1[i - 1] > x1[i]:
                    x1[i] += 2 * np.pi
            x2 = (
                np.mean(x1)
                + np.linspace(-np.pi, np.pi, len(x0), endpoint=False)
                + np.pi / len(x0)
            )
            gf = getfac(x0)
            ratio = (maxfac_inner - 1) / (gf - 1)
            x = x2 * (1 - ratio) + np.array(x1) * (ratio)
            x %= 2 * np.pi
        else:
            while getfac(x) > maxfac_inner:
                extra += 1
                x += fac * (dx(x, 1) + dx(x, -1))
    else:
        while not np.all(dx(x, -1) > 0):
            x += fac * (dx(x, 1) + dx(x, -1))
            extra += 1
        if extra:
            print(f"Required {extra} extra steps in smoothing!")

    if not np.all(dx(x, -1) > 0):
        plt.plot(x, label="result")
        plt.plot(x0, label="init")
        plt.plot(dx(x0, -1))
        plt.plot(dx(x, -1))
        plt.legend()
        plt.title("Not monotonic!")
        plt.show()
        assert False
    return x


def _densify(eq2, exp, plot=False):
    from . import smooth as zs

    pa = 1 - zs.get_pseudoangle(eq2.R, eq2.Z)
    density = np.exp((0.02 + pa) * exp)
    density /= np.sum(density) / len(density)
    densint = np.cumsum(density)
    densint = np.append(densint, np.sum(density))
    points = []
    for i in range(len(densint)):
        while densint[i] > len(points):
            points.append((len(points) - densint[i]) / density[i] + i + 1)
    # assert len(points) == len(density), (len(points), len(density))
    if len(points) == len(density) + 1:
        points = points[:-1]
    assert len(points) == len(density), (
        f"Expected length to be conserved, but found {len(points)} instead of {len(density)}"
    )
    theta = np.array(points)
    theta *= np.pi * 2 / len(density)
    return theta


def grid_elliptic(
    inner,
    outer,
    nx,
    nz,
    show=False,
    tol=1e-10,
    align=True,
    restrict_size=20,
    restrict_factor=2,
    return_coords=False,
    nx_outer=0,
    nx_inner=0,
    legacy_align=False,
    inner_ort=True,
    inner_maxmode=None,
    maxfac_inner=None,
    dz_relax=None,
    guess=None,
    densify_outer=None,
):
    """Create a structured grid between inner and outer boundaries using
     elliptic method

    Coordinates x = x(R, Z) and z = z(R,Z) obey an elliptic equation:

     .. math::

         d^2x/dR^2 + d^2x/dZ^2 = 0

         d^2z/dR^2 + d^2z/dZ^2 = 0

     where here x is in in the domain (0, 1) and z in (0, 2pi)

     The above equations are inverted, giving:

     .. math::

         a*R_xx - 2*b*R_xz + c*R_zz = 0

         a*Z_xx - 2*b*Z_xz + c*Z_zz = 0

     where

     .. math::

         a &= R_z^2 + Z_z^2

         b &= R_z*R_x + Z_x*Z_z

         c &= R_x^2 + Z_x^2

     This is a nonlinear system of equations which is solved
     iteratively.

     Parameters
     ----------
     inner, outer : `RZline`
         Inner and outer boundaries of the domain
     nx : int
         The required radial resolution, including boundaries
     nz : int
         The required poloidal resolution
     show : bool, optional
         Display plots of intermediate results
     tol : float, optional
         Controls when iteration stops
     align : bool, optional
         Attempt to align the inner and outer boundaries
     restrict_size : int, optional
         The size (nx or nz) above which the grid is coarsened
     restrict_factor : int, optional
         The factor by which the grid is divided if coarsened
     nx_outer: int, optional
         The number of additional points outside of the outer boundary
     nx_inner: int, optional
         The number of additional points outside of the inner boundary
     inner_ort: bool, optional
         Whether to place the inner points as close as possible to the
         corresponding outer ones. That increases orthogonality of the grid.
     maxfac_inner : int, optional
         If given, ensure the spacing between smalles and largest distance on
         the inner boundary. Only used with inner_ort. Must be larger than 1.
     return_coords : bool, optional
         If True, return the R, Z coordinates of the grid points,
         instead of a `StructuredPoloidalGrid`

     Returns
     -------
     If return_coords is true, returns R,Z as arrays.
     If return_coords is false, returns a `StructuredPoloidalGrid` object

     References
     ----------
     https://web.archive.org/web/20040530002835/https://www.nada.kth.se/kurser/kth/2D1263/l2.pdf
     https://en.wikipedia.org/wiki/Principles_of_grid_generation

    """
    if nx_outer:
        assert nx_outer > 0
        nx -= nx_outer
    if nx_inner:
        assert nx_inner > 0
        nx -= nx_inner

    assert nx > 1
    assert nz > 1

    # Generate angle values (y coordinate),
    # which should now be equally spaced
    # in distance along inner and outer boundaries
    thetavals = linspace(0, 2 * pi, nz, endpoint=False)
    if densify_outer:
        thetavals = _densify(outer.equallySpaced(n=nz), densify_outer)
    # Radial coordinate
    xvals = linspace(0, 1, nx, endpoint=True)
    # xvals[-1] = 1
    # print(xvals)

    if align and not inner_ort:
        # Align inner and outer boundaries
        # Easiest way is to roll both boundaries
        # so that index 0 is on the outboard midplane.
        # This is useless if inner_ort is used, where the inner points are
        # based on a project of the outer points.
        if len(inner.R) < len(outer.R):
            shorter = inner
            longer = outer
        else:
            shorter = outer
            longer = inner
        ind = np.argmax(shorter.R)
        shorter = rzline.RZline(np.roll(shorter.R, -ind), np.roll(shorter.Z, -ind))
        if legacy_align:
            ind = np.argmax(longer.R)
            longer = rzline.RZline(np.roll(longer.R, -ind), np.roll(longer.Z, -ind))
        else:
            dr = shorter.R - longer.R[:, None]
            dz = shorter.Z - longer.Z[:, None]
            delta = dr**2 + dz**2
            fac = len(longer.R) / len(shorter.R)
            j = np.arange(len(shorter.R), dtype=int)
            sums = [
                np.sum(delta[np.round(j * fac).astype(int) - i, j])
                for i in range(len(longer.R))
            ]
            ind = -np.argmin(sums)
            longer = rzline.RZline(np.roll(longer.R, -ind), np.roll(longer.Z, -ind))
        if len(inner.R) < len(outer.R):
            inner = shorter
            outer = longer
        else:
            outer = shorter
            inner = longer

    def compute_metrics(x, y):
        """
        Compute metric coefficients for the elliptic grid equations.
        Uses central differences in interior, one-sided at boundaries.
        """
        ni, nj = x.shape

        x_xi = np.zeros_like(x)
        x_eta = np.zeros_like(x)
        y_xi = np.zeros_like(y)
        y_eta = np.zeros_like(y)

        # Central differences interior
        x_xi[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
        x_eta[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
        y_xi[1:-1, :] = (y[2:, :] - y[:-2, :]) / 2
        y_eta[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2

        # One-sided at boundaries
        x_xi[0, :] = x[1, :] - x[0, :]
        x_xi[-1, :] = x[-1, :] - x[-2, :]
        x_eta[:, 0] = x[:, 1] - x[:, 0]
        x_eta[:, -1] = x[:, -1] - x[:, -2]
        y_xi[0, :] = y[1, :] - y[0, :]
        y_xi[-1, :] = y[-1, :] - y[-2, :]
        y_eta[:, 0] = y[:, 1] - y[:, 0]
        y_eta[:, -1] = y[:, -1] - y[:, -2]

        alpha = x_eta**2 + y_eta**2
        beta = x_xi * x_eta + y_xi * y_eta
        gamma = x_xi**2 + y_xi**2
        J = x_xi * y_eta - x_eta * y_xi

        return x_xi, x_eta, y_xi, y_eta, alpha, beta, gamma, J

    def compute_control_functions_thomas_middlecoff(x, y):
        """
        Thomas-Middlecoff control functions.
        Computed from boundary conditions to attract grid lines to boundaries.

        P controls clustering in xi direction (along radial lines)
        Q controls clustering in eta direction (along boundary lines)
        """
        ni, nj = x.shape
        x_xi, x_eta, y_xi, y_eta, alpha, beta, gamma, J = compute_metrics(x, y)

        # Second derivatives at boundaries
        x_xixi = np.zeros_like(x)
        y_xixi = np.zeros_like(y)
        x_etaeta = np.zeros_like(x)
        y_etaeta = np.zeros_like(y)

        x_xixi[1:-1, :] = x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]
        y_xixi[1:-1, :] = y[2:, :] - 2 * y[1:-1, :] + y[:-2, :]
        x_etaeta[:, 1:-1] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        y_etaeta[:, 1:-1] = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]

        # Compute P at eta boundaries (j=0 inner, j=-1 outer)
        # P = -(x_xixi * x_xi + y_xixi * y_xi) / (x_xi**2 + y_xi**2)
        denom = x_xi**2 + y_xi**2
        denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

        P_boundary = -(x_xixi * x_xi + y_xixi * y_xi) / denom

        # Compute Q at xi boundaries (i=0, i=-1)
        denom2 = x_eta**2 + y_eta**2
        denom2 = np.where(np.abs(denom2) < 1e-14, 1e-14, denom2)

        Q_boundary = -(x_etaeta * x_eta + y_etaeta * y_eta) / denom2

        # Interpolate P and Q across the domain
        # Linear decay from boundaries into interior
        P = np.zeros((ni, nj))
        Q = np.zeros((ni, nj))

        # P: interpolate from inner (j=0) to outer (j=-1) boundary
        for j in range(nj - 1, 0, -1):
            t = j / (nj - 1)
            P[:, j] = (1 - t) * P_boundary[:, 0] + t * P_boundary[:, -1]

        # Q: interpolate from i=0 to i=-1 boundary
        for i in range(ni):
            t = i / (ni - 1)
            Q[i, :] = (1 - t) * Q_boundary[0, :] + t * Q_boundary[-1, :]

        return P, Q

    def compute_control_functions_steger_sorenson(x, y):
        """
        Steger-Sorenson orthogonality control functions.

        At eta boundaries (j=0 inner, j=-1 outer):
          - Orthogonality condition: beta = 0
          - P is chosen to enforce x_xi . x_eta = 0 at boundary
          - This means grid lines leave the boundary orthogonally

        At xi boundaries (i=0, i=-1):
          - Q is chosen similarly

        In the interior: P and Q are interpolated from boundaries
        with exponential decay so the orthogonality influence is
        strongest at boundaries and decays inward.
        """
        ni, nj = x.shape

        x_xi, x_eta, y_xi, y_eta, alpha, beta, gamma, J = compute_metrics(x, y)

        # ── Second derivatives ────────────────────────────────────────
        x_xixi = np.zeros_like(x)
        y_xixi = np.zeros_like(y)
        x_etaeta = np.zeros_like(x)
        y_etaeta = np.zeros_like(y)

        x_xixi[1:-1, :] = x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]
        y_xixi[1:-1, :] = y[2:, :] - 2 * y[1:-1, :] + y[:-2, :]
        x_etaeta[:, 1:-1] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        y_etaeta[:, 1:-1] = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]

        # Boundaries for second derivatives (one-sided)
        x_xixi[0, :] = x[2, :] - 2 * x[1, :] + x[0, :]
        x_xixi[-1, :] = x[-1, :] - 2 * x[-2, :] + x[-3, :]
        y_xixi[0, :] = y[2, :] - 2 * y[1, :] + y[0, :]
        y_xixi[-1, :] = y[-1, :] - 2 * y[-2, :] + y[-3, :]
        x_etaeta[:, 0] = x[:, 2] - 2 * x[:, 1] + x[:, 0]
        x_etaeta[:, -1] = x[:, -1] - 2 * x[:, -2] + x[:, -3]
        y_etaeta[:, 0] = y[:, 2] - 2 * y[:, 1] + y[:, 0]
        y_etaeta[:, -1] = y[:, -1] - 2 * y[:, -2] + y[:, -3]

        # ── Thomas-Middlecoff attraction terms ────────────────────────
        # These attract grid lines to boundaries
        denom_xi = np.where(np.abs(gamma) < 1e-14, 1e-14, gamma)
        denom_eta = np.where(np.abs(alpha) < 1e-14, 1e-14, alpha)

        P_TM = -(x_xixi * x_xi + y_xixi * y_xi) / denom_xi
        Q_TM = -(x_etaeta * x_eta + y_etaeta * y_eta) / denom_eta

        # ── Steger-Sorenson orthogonality terms ───────────────────────
        # Orthogonality condition at eta=const boundaries (inner/outer):
        #   x_xi . x_eta + y_xi . y_eta = 0  (beta = 0)
        #
        # This gives:
        #   P = -(alpha * x_xixi + ... ) / (J^2 * gamma / alpha)
        #   at the boundary
        #
        # More precisely, at j=0 and j=-1:
        #   P_orth = -beta / gamma * (some geometric factor)

        J_safe = np.where(np.abs(J) < 1e-14, 1e-14, J)

        # Orthogonality correction:
        # Force beta=0 at boundaries by computing the P,Q needed
        # P_orth forces x_eta to be perpendicular to x_xi at eta boundaries
        P_orth = np.zeros_like(x)
        Q_orth = np.zeros_like(x)

        # At j=0 (inner boundary): enforce orthogonality
        # P = -1/gamma * (x_xi*x_xieta + y_xi*y_xieta) at j=0
        # Approximate x_xieta at j=0 using one-sided difference
        x_xieta_j0 = (
            (x[1:, 1] - x[:-1, 1])  # xi-deriv at j=1
            - (x[1:, 0] - x[:-1, 0])
        )  # xi-deriv at j=0
        y_xieta_j0 = (y[1:, 1] - y[:-1, 1]) - (y[1:, 0] - y[:-1, 0])

        # Pad to full size (periodic)
        x_xieta_j0 = np.concatenate([x_xieta_j0, [x_xieta_j0[0]]])
        y_xieta_j0 = np.concatenate([y_xieta_j0, [y_xieta_j0[0]]])

        P_orth[:, 0] = (
            -(x_xi[:, 0] * x_xieta_j0 + y_xi[:, 0] * y_xieta_j0) / denom_xi[:, 0]
        )

        # At j=-1 (outer boundary): enforce orthogonality
        x_xieta_jN = (x[1:, -1] - x[:-1, -1]) - (x[1:, -2] - x[:-1, -2])
        y_xieta_jN = (y[1:, -1] - y[:-1, -1]) - (y[1:, -2] - y[:-1, -2])

        x_xieta_jN = np.concatenate([x_xieta_jN, [x_xieta_jN[0]]])
        y_xieta_jN = np.concatenate([y_xieta_jN, [y_xieta_jN[0]]])

        P_orth[:, -1] = (
            -(x_xi[:, -1] * x_xieta_jN + y_xi[:, -1] * y_xieta_jN) / denom_xi[:, -1]
        )

        # ── Combine TM attraction + SS orthogonality ──────────────────
        # At boundaries: use full orthogonality correction
        # In interior: interpolate with exponential decay from boundaries
        P_bound = np.zeros_like(x)
        Q_bound = np.zeros_like(x)

        # Blend TM and SS at boundaries
        # alpha_blend=1: pure orthogonality, 0: pure TM attraction
        alpha_blend = 0.7  # tunable: higher = more orthogonal

        P_bound[:, 0] = alpha_blend * P_orth[:, 0] + (1 - alpha_blend) * P_TM[:, 0]
        P_bound[:, -1] = alpha_blend * P_orth[:, -1] + (1 - alpha_blend) * P_TM[:, -1]

        Q_bound[0, :] = Q_TM[0, :]
        Q_bound[-1, :] = Q_TM[-1, :]

        # ── Interpolate P and Q into interior ─────────────────────────
        # Exponential decay from boundaries so orthogonality is enforced
        # strongly at boundary and smoothly relaxes inward
        P = np.zeros((ni, nj))
        Q = np.zeros((ni, nj))

        # Decay length scale (tune this: smaller = faster decay inward)
        decay = 3.0

        eta = np.linspace(0, 1, nj)

        # Exponential weights from each boundary
        w_inner = np.exp(-decay * eta)  # strong at j=0
        w_outer = np.exp(-decay * (1 - eta))  # strong at j=-1

        # Normalize so weights sum to 1 where both are significant
        w_sum = w_inner + w_outer
        w_sum = np.where(w_sum < 1e-14, 1e-14, w_sum)

        for i in range(ni):
            P[i, :] = (w_inner * P_bound[i, 0] + w_outer * P_bound[i, -1]) / w_sum

        xi = np.linspace(0, 1, ni)
        w_left = np.exp(-decay * xi)
        w_right = np.exp(-decay * (1 - xi))
        w_sum2 = w_left + w_right
        w_sum2 = np.where(w_sum2 < 1e-14, 1e-14, w_sum2)

        for j in range(nj):
            Q[:, j] = (w_left * Q_bound[0, j] + w_right * Q_bound[-1, j]) / w_sum2

        return P, Q

    def solve_elliptic_grid_TTM(
        x_init,
        y_init,
        n_iter=500,
        omega=1.5,
        recompute_control_every=50,
        orthogonality=True,
    ):
        """
        Solve elliptic grid generation equations with
        Steger-Sorenson orthogonality + Thomas-Middlecoff attraction.

        Grid layout:
          xi  (i) : along boundary (0..ni-1), PERIODIC
          eta (j) : radial direction (0=inner, nj-1=outer), FIXED
        """
        x = x_init.copy()
        y = y_init.copy()
        ni, nj = x.shape

        print(f"Grid size: {ni} x {nj}")
        print(f"Orthogonality correction: {orthogonality}")

        for iteration in range(n_iter):
            if iteration % recompute_control_every == 0:
                if orthogonality:
                    P, Q = compute_control_functions_steger_sorenson(x, y)
                else:
                    P, Q = compute_control_functions_thomas_middlecoff(x, y)

            x_old = x.copy()
            y_old = y.copy()

            for j in range(1, nj - 1):
                for i in range(ni):
                    im = (i - 1) % ni
                    ip = (i + 1) % ni

                    x_xi = (x[ip, j] - x[im, j]) / 2
                    x_eta = (x[i, j + 1] - x[i, j - 1]) / 2
                    y_xi = (y[ip, j] - y[im, j]) / 2
                    y_eta = (y[i, j + 1] - y[i, j - 1]) / 2

                    alpha = x_eta**2 + y_eta**2
                    beta = x_xi * x_eta + y_xi * y_eta
                    gamma = x_xi**2 + y_xi**2
                    J = x_xi * y_eta - x_eta * y_xi

                    x_xieta = (
                        x[ip, j + 1] - x[im, j + 1] - x[ip, j - 1] + x[im, j - 1]
                    ) / 4
                    y_xieta = (
                        y[ip, j + 1] - y[im, j + 1] - y[ip, j - 1] + y[im, j - 1]
                    ) / 4

                    rhs_x = 2 * beta * x_xieta - J**2 * (
                        P[i, j] * x_xi + Q[i, j] * x_eta
                    )
                    rhs_y = 2 * beta * y_xieta - J**2 * (
                        P[i, j] * y_xi + Q[i, j] * y_eta
                    )

                    denom = 2 * (alpha + gamma)
                    if abs(denom) < 1e-14:
                        continue

                    x_new = (
                        alpha * (x[im, j] + x[ip, j])
                        + gamma * (x[i, j - 1] + x[i, j + 1])
                        + rhs_x
                    ) / denom
                    y_new = (
                        alpha * (y[im, j] + y[ip, j])
                        + gamma * (y[i, j - 1] + y[i, j + 1])
                        + rhs_y
                    ) / denom

                    x[i, j] = omega * x_new + (1 - omega) * x_old[i, j]
                    y[i, j] = omega * y_new + (1 - omega) * y_old[i, j]

            dx = np.max(np.abs(x - x_old))
            dy = np.max(np.abs(y - y_old))
            residual = max(dx, dy)

            if iteration % 50 == 0:
                _, _, _, _, _, _, _, J_field = compute_metrics(x, y)
                min_J = np.min(J_field[1:-1, 1:-1])
                print(
                    f"  iter {iteration:4d}: residual={residual:.2e}, "
                    f"min(J)={min_J:.6f}"
                )
                if min_J < 0:
                    print("  WARNING: negative Jacobian - mesh folding!")

            if residual < 1e-10:
                print(f"Converged at iteration {iteration}")
                break

        return x, y

    def solve_equation_new(R, Z, dz_relax, dx, dz, plotonfailure=True):
        x_smooth, y_smooth = solve_elliptic_grid_TTM(
            R.T,
            Z.T,
            n_iter=200,
            omega=1.5,
            recompute_control_every=50,
            orthogonality=False,  # ← enables Steger-Sorenson
        )
        plt.plot(x_smooth, y_smooth)
        plt.plot(x_smooth.T, y_smooth.T)
        plt.show(block=True)
        return x_smooth.T, y_smooth.T

    # Start solver loop
    def solve_equation(R, Z, dz_relax, dx, dz, plotonfailure=True):
        allgood = _allGood(R, Z)
        maxit = max(max(R.shape) * 10, 300) * 10
        for _ in range(maxit):
            # Calculate coefficients, which exclude boundary points
            # Note that the domain is periodic in y so roll arrays

            R_xm = R[:-2, :]  # R(x-1,z)
            R_xp = R[2:, :]  # R(x+1,z)
            R_zm = np.roll(R, 1, axis=1)  # R(x, z-1)
            R_zp = np.roll(R, -1, axis=1)  # R(x, z+1)
            R_xmzm = R_zm[:-2, :]  # R(x-1, z-1)
            R_xpzm = R_zm[2:, :]  # R(x+1, z-1)
            R_xmzp = R_zp[:-2, :]  # R(x-1, z+1)
            R_xpzp = R_zp[2:, :]  # R(x+1, z+1)
            R_zm = R_zm[1:-1, :]  # Now chop off x boundaries
            R_zp = R_zp[1:-1, :]  # This is done to minimise number of rolls

            Z_xm = Z[:-2, :]
            Z_xp = Z[2:, :]
            Z_zm = np.roll(Z, 1, axis=1)
            Z_zp = np.roll(Z, -1, axis=1)
            Z_xmzm = Z_zm[:-2, :]
            Z_xpzm = Z_zm[2:, :]
            Z_xmzp = Z_zp[:-2, :]
            Z_xpzp = Z_zp[2:, :]
            Z_zm = Z_zm[1:-1, :]
            Z_zp = Z_zp[1:-1, :]

            dz_relax = dz_relax or 10 / 3
            dRdz = dz_relax * (R_zp - R_zm) / (2.0 * dz)
            dRdx = (R_xp - R_xm) / (2.0 * dx)

            dZdz = dz_relax * (Z_zp - Z_zm) / (2.0 * dz)
            dZdx = (Z_xp - Z_xm) / (2.0 * dx)

            a = dRdz**2 + dZdz**2
            b = dRdz * dRdx + dZdx * dZdz
            c = dRdx**2 + dZdx**2

            # Now solve a*R_xx - 2*b*R_xz + c*R_zz = 0
            # For now using Jacobi update

            a_dx2 = a / dx**2
            b_dxdz = b / (2.0 * dx * dz)
            c_dz2 = c / dz**2
            inv_diag = 1.0 / (2 * a / dx**2 + 2 * c / dz**2)

            Rold = R.copy()
            Zold = Z.copy()

            R[1:-1, :] = (
                a_dx2 * (R_xm + R_xp)
                - b_dxdz * (R_xpzp - R_xmzp - R_xpzm + R_xmzm)
                + c_dz2 * (R_zm + R_zp)
            ) * inv_diag

            Z[1:-1, :] = (
                a_dx2 * (Z_xm + Z_xp)
                - b_dxdz * (Z_xpzp - Z_xmzp - Z_xpzm + Z_xmzm)
                + c_dz2 * (Z_zm + Z_zp)
            ) * inv_diag
            maxchange_sq = np.amax((R - Rold) ** 2 + (Z - Zold) ** 2)
            if False:  # and not allgood.checkOuter(R, Z):
                if not allgood.checkOuter(Rold, Zold):
                    plt.figure()
                    plt.plot(Rold, Zold)
                    plt.plot(Rold.T, Zold.T)
                    plt.title("old")
                    plt.gca().set_aspect(1)
                    plt.show(block=True)
                    assert False

                lo = 0
                hi = 1
                for _ in range(10):
                    mid = (lo + hi) / 2
                    maybeR = R * mid + Rold * (1 - mid)
                    maybeZ = Z * mid + Zold * (1 - mid)
                    if allgood.checkOuter(maybeR, maybeZ):
                        lo = mid
                    else:
                        hi = mid
                mid = lo
                mid /= 2
                if mid < 0.001:
                    for Ri, Zi, lbl in ((Rold, Zold, "old"), (R, Z, "proposed")):
                        plt.figure()
                        plt.plot(Ri, Zi)
                        plt.plot(Ri.T, Zi.T)
                        plt.title(lbl)
                        plt.gca().set_aspect(1)
                    plt.show(block=True)
                R = R * mid + Rold * (1 - mid)
                Z = Z * mid + Zold * (1 - mid)
                print(f"Only stepping {mid:.6f}")

            # R, Z = _ensure_monotonic(R, Z, overshoot=1.1)

            if maxchange_sq < tol:
                return R, Z

        else:
            print(
                f"Failed to converge - grid shape {R.shape}: Error was {maxchange_sq} > {tol} ; {maxit} iterations"
            )
            if plotonfailure:
                print("Convergence failure, trying to plot ...")
                if plotting_available:
                    if not show:
                        plt.figure()
                        # Markers on original points on inner and outer boundaries
                        plt.plot(inner.R, inner.Z, "-o", label="inner")
                        plt.plot(outer.R, outer.Z, "-o", label="outer")

                        # Black lines through inner and outer boundaries
                        r, z = inner.position(np.linspace(0, 2 * np.pi, 10 * nz))
                        plt.plot(r, z, "k")
                        r, z = outer.position(np.linspace(0, 2 * np.pi, 10 * nz))
                        plt.plot(r, z, "k")

                        # Red dots to mark the inner and outer boundaries
                        plt.plot(R[0, :], Z[0, :], "ro")
                        plt.plot(R[-1, :], Z[-1, :], "ro")

                        plt.plot(R, Z, "k-")
                        plt.plot(R.T, Z.T, "g-")
                    plt.show()

            raise RuntimeError(
                f"Failed to converge - grid shape {R.shape}: Error was {maxchange_sq} > {tol}"
            )

    def fft_smooth(x, maxmode=None):
        dx = x - np.roll(x, 1)
        dx[dx < -np.pi] += np.pi * 2
        dx[dx > np.pi] -= np.pi * 2
        f = np.fft.fft(dx)
        n = len(f)
        nh = n // 2
        nq = maxmode
        f[nq:nh] = 0
        f[nh + 1 : -nq] = 0
        dxn = np.real(np.fft.ifft(f))
        xnew = np.empty_like(x)
        xnew[0] = x[0]
        for i in range(len(x) - 1):
            xnew[i + 1] = xnew[i] + dxn[i]
        return xnew % (np.pi * 2)

    if (nx > restrict_size) or (nz > restrict_size):
        # Create a coarse grid first to get a starting guess
        # Only restrict the dimensions which exceed restrict_size
        # Note that this might result in multiple levels of resolution

        nx_r = nx
        if nx > restrict_size:
            nx_r = int(nx / restrict_factor)

        nz_r = nz
        if nz > restrict_size:
            nz_r = int(nz / restrict_factor)

        # Create the coarse mesh
        R_r, Z_r = grid_elliptic(
            inner,
            outer,
            nx_r,
            nz_r,
            align=False,
            tol=tol,
            restrict_size=restrict_size,
            restrict_factor=restrict_factor,
            return_coords=True,
            inner_ort=inner_ort,
            inner_maxmode=inner_maxmode,
            maxfac_inner=maxfac_inner,
            dz_relax=dz_relax,
        )

        # Note: Lower case x,z are indices
        z_r = linspace(
            0, 2 * pi, nz_r + 1, endpoint=True
        )  # Add on the final point duplicating the first
        x_r = linspace(0, 1.0, nx_r, endpoint=True)
        # x_r[0] = 0

        # Note: Upper case R,Z are real-space locations
        R_r = np.concatenate((R_r, np.reshape(R_r[:, 0], (nx_r, 1))), axis=1)
        Z_r = np.concatenate((Z_r, np.reshape(Z_r[:, 0], (nx_r, 1))), axis=1)

        # Now interpolate
        print("Interpolating!")
        spl = RectBivariateSpline(x_r, z_r, R_r)
        R = spl(xvals, thetavals, grid=True)
        spl = RectBivariateSpline(x_r, z_r, Z_r)
        Z = spl(xvals, thetavals, grid=True)

        # Make sure that the inner and outer boundaries are on the
        # inner and outer RZline, not interpolated
        R[-1, :] = outer.Rvalue(thetavals)
        Z[-1, :] = outer.Zvalue(thetavals)

        if inner_ort:
            thetavals_inner = [inner.closestPoint(*x) for x in zip(R[-1], Z[-1])]
            thetavals_inner = _laplace(thetavals_inner, maxfac_inner)
            if inner_maxmode:
                thetavals_inner = fft_smooth(thetavals_inner, inner_maxmode)
        else:
            thetavals_inner = thetavals

        R[0, :] = inner.Rvalue(thetavals_inner)
        Z[0, :] = inner.Zvalue(thetavals_inner)
        # R, Z = _ensure_monotonic(R, Z, overshoot=1.1)

    elif guess is not None:
        R, Z = guess
    else:
        # Interpolate coordinates of inner and outer boundary
        from . import smooth

        pa = smooth.get_pseudoangle(outer.Rvalue(thetavals), outer.Zvalue(thetavals), 3)

        Router = outer.Rvalue(thetavals)
        Zouter = outer.Zvalue(thetavals)

        if inner_ort:
            thetavals_inner = [inner.closestPoint(*x) for x in zip(Router, Zouter)]
            thetavals_inner = _laplace(thetavals_inner, maxfac_inner)
            if inner_maxmode:
                thetavals_inner = fft_smooth(thetavals_inner, inner_maxmode)
        else:
            thetavals_inner = thetavals

        Rinner = inner.Rvalue(thetavals_inner)
        Zinner = inner.Zvalue(thetavals_inner)
        R, Z = _get_guess(
            np.array((Router, Zouter)), np.array((Rinner, Zinner)), nx, nz
        )

        # # Interpolate in x between inner and outer
        # # to get starting guess for a grid
        # # R = xvals[:, None] * Router[None, :] + (1 - xvals[:, None]) * Rinner[None, :]
        # # Z = xvals[:, None] * Zouter[None, :] + (1 - xvals[:, None]) * Zinner[None, :]
        # def interpolate(xvals, inner, outer, fft=True, smoothfac=1):

        #     if not fft:

        #         def smooth(values, smoothfac):
        #             fac = np.fft.fft(values)
        #             n = len(values)
        #             nh = n // 2
        #             nq = 5
        #             fac[nq:nh] *= smoothfac
        #             fac[nh + 1 : -nq] *= smoothfac
        #             return np.real(np.fft.ifft(fac))

        #         Ro = smooth(outer, smoothfac)
        #         Ri = smooth(inner, smoothfac)
        #         print(xvals, Ro.mean(), Ri.mean(), outer.mean(), inner.mean())
        #         return xvals[:, None] * Ro[None, :] + (1 - xvals[:, None]) * Ri[None, :]
        #     inf = np.fft.fft(inner)
        #     outf = np.fft.fft(outer)
        #     n = len(inner)
        #     nh = n // 2
        #     nq = 5
        #     inf[nq:nh] *= smoothfac
        #     inf[nh + 1 : -nq] *= smoothfac
        #     outf[nq:nh] *= smoothfac
        #     outf[nh + 1 : -nq] *= smoothfac

        #     f = xvals[:, None] * outf[None, :] + (1 - xvals[:, None]) * inf[None, :]
        #     plt.figure()
        #     plt.plot(inf, label="inf")
        #     plt.plot(outf, label="outf")
        #     plt.plot(inf, label="inf")
        #     plt.plot(outf, label="outf")
        #     # plt.plot(f, label=f)
        #     plt.show(block=True)
        #     return np.real(np.fft.ifft(f))

        # xvals = linspace(0, 1, nx, endpoint=True)
        # xvals[-1] = 1
        # usefft = False
        # sf = 1
        # working = 0
        # reset = 0.9
        # lastgood = None
        # while True:
        #     print(sf)
        #     R, Z = _get_guess(
        #         def _get_guess(target_outer, target_inner, nx, nz):

        #     R = interpolate(xvals, Rinner, Router, usefft, sf)
        #     Z = interpolate(xvals, Zinner, Zouter, usefft, sf)
        #     if lastgood is not None:
        #         R[1:-1] = lastgood[0][1:-1]
        #         Z[1:-1] = lastgood[1][1:-1]
        #         # assert False
        #     R, Z = _ensure_monotonic_force(R, Z)
        #     plt.figure()
        #     plt.plot(R, Z)
        #     plt.plot(R.T, Z.T)
        #     plt.show(block=True)
        #     try:
        # R, Z = _ensure_monotonic(R, Z)
        R, Z = solve_equation(
            R,
            Z,
            dz_relax,
            xvals[1] - xvals[0],
            thetavals[1] - thetavals[0],
            True,
        )
        # except RuntimeError:
        #     # raise
        #     failed = sf
        #     sf = (sf + working) / 2
        # else:
        #     if sf == 1:
        #         break
        #     working_ = sf
        #     sf = sf * 2 - working
        #     working = working_
        #     lastgood = R, Z
        #     if sf > reset:
        #         sf = 1
        #         reset = (reset + 1) / 2
        # print(sf, failed, working, reset)

    # R, Z = _ensure_monotonic(R, Z, overshoot=1.1)
    dz = thetavals[1] - thetavals[0]

    if show and plotting_available:
        # Markers on original points on inner and outer boundaries
        plt.plot(inner.R, inner.Z, "-o")
        plt.plot(outer.R, outer.Z, "-o")

        # Black lines through inner and outer boundaries
        r, z = inner.position(np.linspace(0, 2 * np.pi, 10 * nz))
        plt.plot(r, z, "k")
        r, z = outer.position(np.linspace(0, 2 * np.pi, 10 * nz))
        plt.plot(r, z, "k")

        # Red dots to mark the inner and outer boundaries
        plt.plot(R[0, :], Z[0, :], "ro")
        plt.plot(R[-1, :], Z[-1, :], "ro")

    R, Z = solve_equation(
        R, Z, dz_relax, xvals[1] - xvals[0], thetavals[1] - thetavals[0]
    )

    if nx_outer:
        dn = np.arange(nx_outer) + 1
        dR = R[-1] - R[-2]
        dZ = Z[-1] - Z[-2]
        R = np.vstack([R, R[-1] + dR[None, :] * dn[:, None]])
        Z = np.vstack([Z, Z[-1] + dZ[None, :] * dn[:, None]])
    if nx_inner:
        dn = np.arange(nx_inner)[::-1] + 1
        dR = R[0] - R[1]
        dZ = Z[0] - Z[1]
        R = np.vstack([R[0] + dR[None, :] * dn[:, None], R])
        Z = np.vstack([Z[0] + dZ[None, :] * dn[:, None], Z])

    if show and plotting_available:
        plt.plot(R, Z)
        plt.plot(np.transpose(R), np.transpose(Z))
        plt.show()

    if return_coords:
        return R, Z
    return StructuredPoloidalGrid(R, Z)


if __name__ == "__main__":
    inner = rzline.shaped_line(R0=3.0, a=0.5, elong=1.0, triang=0.0, indent=1.0, n=50)
    outer = rzline.shaped_line(R0=2.8, a=1.5, elong=1.0, triang=0.0, indent=0.2, n=50)

    grid = grid_elliptic(inner, outer, 100, 100, show=True)

    x, z = grid.findIndex([2.0, 1.9], [1.5, 2.0])
    print(x, z)
