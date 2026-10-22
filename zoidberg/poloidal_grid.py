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
`grid_inverse_harmonic`
    Create a `StructuredPoloidalGrid` from inner and outer RZLine
    objects as a smooth spline map whose inverse is harmonic

"""

import warnings
from types import SimpleNamespace

import numpy as np
import shapely
from numpy import linspace, pi, zeros
from scipy.interpolate import BSpline, CubicSpline, RectBivariateSpline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.spatial import cKDTree as KDTree

try:
    import matplotlib.pyplot as plt

    plotting_available = True
except ImportError:
    warnings.warn("Couldn't import matplotlib, plotting not available.")
    plotting_available = False

from . import rzline


class PoloidalGrid:
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
        return f"RectangularPoloidalGrid({self.nx},{self.nz},{self.Lx},{self.Lz},Rcentre={self.Rcentre},Zcentre={self.Zcentre})"

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

        _dists, ind = self.tree.query(position)

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

    # Radial coordinate
    xvals = linspace(0, 1.0, nx, endpoint=True)

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

    def laplace(x):
        def dx(x, sign):
            dxv = np.roll(x, sign) - x
            dxv = np.remainder(dxv + np.pi, 2 * np.pi) - np.pi
            return dxv

        fac = 0.1 * nz / 192
        steps = 10
        if fac > 0.1:
            steps = int(steps * fac / 0.1)
            fac = 0.1
        x0 = x.copy()
        for i in range(steps):
            x += fac * (dx(x, 1) + dx(x, -1))

        extra = 0
        if maxfac_inner:

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

        # Note: Upper case R,Z are real-space locations
        R_r = np.concatenate((R_r, np.reshape(R_r[:, 0], (nx_r, 1))), axis=1)
        Z_r = np.concatenate((Z_r, np.reshape(Z_r[:, 0], (nx_r, 1))), axis=1)

        # Now interpolate
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
            thetavals_inner = laplace(thetavals_inner)
            if inner_maxmode:
                thetavals_inner = fft_smooth(thetavals_inner, inner_maxmode)
        else:
            thetavals_inner = thetavals

        R[0, :] = inner.Rvalue(thetavals_inner)
        Z[0, :] = inner.Zvalue(thetavals_inner)

    else:
        # Interpolate coordinates of inner and outer boundary
        Router = outer.Rvalue(thetavals)
        Zouter = outer.Zvalue(thetavals)

        if inner_ort:
            thetavals_inner = [inner.closestPoint(*x) for x in zip(Router, Zouter)]
            thetavals_inner = laplace(thetavals_inner)
            if inner_maxmode:
                thetavals_inner = fft_smooth(thetavals_inner, inner_maxmode)
        else:
            thetavals_inner = thetavals

        Rinner = inner.Rvalue(thetavals_inner)
        Zinner = inner.Zvalue(thetavals_inner)

        # Interpolate in x between inner and outer
        # to get starting guess for a grid
        R = xvals[:, None] * Router[None, :] + (1 - xvals[:, None]) * Rinner[None, :]
        Z = xvals[:, None] * Zouter[None, :] + (1 - xvals[:, None]) * Zinner[None, :]

    dx = xvals[1] - xvals[0]
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

    # Start solver loop
    maxit = max(max(R.shape) * 10, 300)
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

        if maxchange_sq < tol:
            break

    else:
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
            plt.show()

        raise RuntimeError(
            f"Failed to converge - grid shape {R.shape}: Error was {maxchange_sq} > {tol}"
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


def grid_inverse_harmonic(
    inner,
    outer,
    nx,
    nz,
    show=False,
    tol=1e-10,
    align=True,
    *,
    return_coords=False,
    nx_outer=0,
    nx_inner=0,
    degree=3,
    resolution=1,
    homogenize=1.0,
    maxiter=50,
    eps=1e-4,
    max_refine=1,
    verbose=False,
    **elliptic_options,
):
    """Create a structured grid between inner and outer boundaries as
    the inverse of a harmonic map, discretised with smooth splines

    This follows J. Hinz and A. Buffa, "On the use of elliptic PDEs for
    the parameterisation of planar multipatch domains", Engineering with
    Computers 40 (2024) 3735, adapted to the annular (doubly connected)
    domains used by zoidberg.

    The map (x, t) -> (R, Z) is sought such that its inverse, i.e. the
    grid coordinates x(R, Z) and t(R, Z), are harmonic in the physical
    domain:

    .. math::

        \\nabla \\cdot (\\omega \\nabla x) = 0, \\quad
        \\nabla \\cdot (\\omega \\nabla t) = 0

    with x = 0 on the inner and x = 1 on the outer boundary and t
    periodic. The boundary points are equally spaced in arc length.

    For omega = 1 (``homogenize=0``) the exact solution is free of folds:
    in conformal coordinates (s, phi) of the annulus, x is linear in s
    and dt/dphi is a periodic harmonic function that is positive on both
    boundaries, hence positive everywhere by the minimum principle. This
    assumes nested boundaries with C^(1,alpha) regularity (e.g. the
    default cubic spline RZlines) and a strictly monotone C^1 boundary
    correspondence, and holds for the exact, not the discrete map.

    On strongly shaped boundaries this map has very large cells in
    protrusions and very small ones in concave regions, which at a finite
    resolution can give non-convex cells. Therefore by default
    omega = det(J)^(-1), which homogenises the cell sizes (section 4.3 of
    the paper, method 2). No such theorem is known for omega != 1; the
    validity of the discrete map is ensured by the checks below.

    The map is represented by a single tensor-product B-spline of
    `degree` p, clamped in x and periodic in t. With maximal regularity
    the map is C^(p-1) everywhere, so the metric coefficients are
    C^(p-2), e.g. continuously differentiable for the default p = 3.

    The solve follows section 3 of the paper: a fixed-point iteration on
    the nondivergence form A : H(R) = A : H(Z) = 0, tested with the
    Laplacian of the test functions, provides the initial guess for a
    Newton iteration on the regularised weak form, whose 1/det(J)
    barrier keeps the discrete map from folding. The Jacobian
    determinant is then checked on a dense sample of every element; if
    it is not positive the spline is refined and the solve repeated.

    Parameters
    ----------
    inner, outer : `RZline`
        Inner and outer boundaries of the domain
    nx : int
        The required radial resolution, including boundaries
    nz : int
        The required poloidal resolution
    show : bool, optional
        Plot the resulting grid
    tol : float, optional
        Newton iteration stops when the largest update of a spline
        coefficient (in units of the square root of the domain area) is
        below `tol`
    return_coords : bool, optional
        If True, return the R, Z coordinates of the grid points,
        instead of a `StructuredPoloidalGrid`
    nx_outer : int, optional
        The number of additional points outside of the outer boundary.
        These continue the map with its second order Taylor expansion at
        the boundary. A warning is given if any of these cells fold.
    nx_inner : int, optional
        The number of additional points outside of the inner boundary
    align : bool, optional
        Shift the poloidal starting point of the inner boundary so that
        inner and outer boundary points are as close as possible
    degree : int, optional
        Polynomial degree of the spline map (>= 3 for differentiable
        metric coefficients)
    resolution : int, optional
        Number of spline elements per grid cell in each direction. Must
        be an integer, so that the output points on the boundaries are
        interpolation points of the spline and lie exactly on the lines.
    homogenize : float, optional
        Exponent k >= 0 of the diffusivity omega = det(J)^(-k), which
        contracts large and expands small cells (section 4.3 of the
        paper, method 2). 0 gives the plain inverse harmonic map.
    maxiter : int, optional
        Maximum number of Newton iterations
    eps : float, optional
        Regularisation of det(J) in the weak form (eq. 38 of the paper),
        relative to the mean of det(J). Should be small, e.g. 1e-6 to 1e-3.
    max_refine : int, optional
        How often the spline may be refined if the map folds. Each
        refinement needs about four times the memory and run time.
    verbose : bool, optional
        Print the convergence history
    restrict_size, restrict_factor, legacy_align, inner_ort,
    inner_maxmode, maxfac_inner, dz_relax : optional
        Options of `grid_elliptic` which have no meaning here. They are
        accepted, so that this function can replace `grid_elliptic`, but
        ignored with a warning.

    Returns
    -------
    If return_coords is true, returns R,Z as arrays.
    If return_coords is false, returns a `StructuredPoloidalGrid` object

    """
    ignored = {
        "restrict_size",
        "restrict_factor",
        "legacy_align",
        "inner_ort",
        "inner_maxmode",
        "maxfac_inner",
        "dz_relax",
    }
    unknown = set(elliptic_options) - ignored
    if unknown:
        raise TypeError(
            f"grid_inverse_harmonic: unexpected arguments {sorted(unknown)}"
        )
    if elliptic_options:
        warnings.warn(
            "grid_inverse_harmonic: ignoring grid_elliptic options "
            f"{sorted(elliptic_options)}"
        )

    def check(condition, message):
        if not condition:
            raise ValueError(f"grid_inverse_harmonic: {message}")

    check(nx_outer >= 0 and nx_inner >= 0, "nx_outer and nx_inner must not be negative")
    nx -= nx_outer + nx_inner
    check(nx > 1, "need at least two points between the boundaries")
    check(degree >= 2, "degree must be at least 2")
    check(
        int(resolution) == resolution and resolution >= 1,
        "resolution must be a positive integer",
    )
    resolution = int(resolution)
    check(resolution * nz >= degree + 1, "nz is too small for the spline degree")
    check(homogenize >= 0, "homogenize must not be negative")
    check(eps > 0, "eps must be positive")
    p = degree

    # Both lines must be simple closed curves, the inner inside the outer
    polygons = []
    for name, line in (("inner", inner), ("outer", outer)):
        polygon = shapely.Polygon(zip(*line.position(linspace(0, 2 * pi, 8 * nz))))
        check(polygon.is_valid, f"the {name} line intersects itself")
        polygons.append(polygon)
    check(
        polygons[1].contains(polygons[0]),
        "the inner line must lie inside the outer line",
    )

    def arclength_inverse(line):
        """Return function mapping arc length fraction u to line theta"""
        nfine = max(64 * len(line.R), 4096)
        theta = linspace(0, 2 * pi, nfine, endpoint=False)
        speed = np.hypot(line.Rvalue(theta, deriv=1), line.Zvalue(theta, deriv=1))
        if np.amin(speed) <= 0.0:
            raise ValueError("Boundary line has a vanishing tangent")
        speed = CubicSpline(
            np.append(theta, 2 * pi), np.append(speed, speed[0]), bc_type="periodic"
        )
        length = speed.antiderivative()
        theta_ext = np.append(theta, 2 * pi)
        u = (length(theta_ext) - length(0.0)) / (length(2 * pi) - length(0.0))
        # theta - 2 pi u is periodic in u, so it can be a periodic spline
        offset = CubicSpline(u, theta_ext - 2 * pi * u, bc_type="periodic")
        # The lines have to go anticlockwise, as the grid assumes this
        R, Z = line.position(theta)
        sign = 1 if np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z) > 0 else -1

        def theta_of(v):
            v = sign * np.asarray(v)
            return 2 * pi * v + offset(np.remainder(v, 1.0))

        return theta_of

    def cardinal(s, deriv=0):
        """Uniform cardinal B-spline of degree p on [0, p+1]"""
        spl = BSpline.basis_element(np.arange(p + 2), extrapolate=False)
        if deriv:
            spl = spl.derivative(deriv)
        return np.nan_to_num(spl(s), nan=0.0)

    def periodic_matrix(t, nt, deriv=0):
        """Dense matrix of periodic basis functions (or derivatives) at t"""
        s = np.remainder(
            nt * np.asarray(t)[:, None] - np.arange(nt)[None, :] + (p + 1) / 2, nt
        )
        return cardinal(s, deriv) * nt**deriv

    def clamped_knots(nex):
        return np.concatenate((np.zeros(p), linspace(0, 1, nex + 1), np.ones(p)))

    def clamped_matrix(knots, x, deriv=0):
        """Dense matrix of clamped basis functions; extrapolates outside [0,1]"""
        nr = len(knots) - p - 1
        out = np.empty((len(x), nr))
        for i in range(nr):
            coef = np.zeros(nr)
            coef[i] = 1.0
            spl = BSpline(knots, coef, p, extrapolate=True)
            out[:, i] = spl(x, nu=deriv)
        return out

    # Boundary correspondence: equal arc length on both lines
    inner_theta = arclength_inverse(inner)
    outer_theta = arclength_inverse(outer)

    shift = 0.0
    if align:
        # Minimise sum |inner(u + shift) - outer(u)|^2 using a circular
        # cross correlation of the complex positions R + iZ
        nsample = 16 * max(nz, len(inner.R), len(outer.R))
        u = np.arange(nsample) / nsample
        zin = inner.Rvalue(inner_theta(u)) + 1j * inner.Zvalue(inner_theta(u))
        zout = outer.Rvalue(outer_theta(u)) + 1j * outer.Zvalue(outer_theta(u))
        corr = np.fft.ifft(np.fft.fft(zin) * np.conj(np.fft.fft(zout)))
        shift = np.argmax(np.real(corr)) / nsample

    # Normalise coordinates to unit area, so that eps is scale invariant
    centre = np.mean(outer.R) + 1j * np.mean(outer.Z)
    scale = np.sqrt(polygons[1].area - polygons[0].area)

    # Gauss-Legendre quadrature on [0, 1]
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(p + 1)
    gauss_x = 0.5 * (gauss_x + 1)
    gauss_w = 0.5 * gauss_w
    nq = p + 1

    def make_space(nex, nt):
        """Spline space with nex elements in x and nt periodic functions in t"""
        S = SimpleNamespace(nex=nex, nt=nt)
        S.knots = clamped_knots(nex)
        S.nr = nr = nex + p  # Basis functions in x
        S.ndof = nr * nt

        # Basis functions on all quadrature points in x.
        # Shape (nex, nq, p+1): only the p+1 functions nonzero per element
        xq = (np.arange(nex)[:, None] + gauss_x[None, :]).ravel() / nex
        S.Bx = [
            np.stack([b[e * nq : (e + 1) * nq, e : e + p + 1] for e in range(nex)])
            for b in (clamped_matrix(S.knots, xq, d) for d in range(3))
        ]

        # Periodic basis is shift invariant, so all elements are identical.
        # Element e spans nt*t in [e + o, e + o + 1]
        o = 0.0 if p % 2 else 0.5
        first = round(o + 1 - (p + 1) / 2)  # First nonzero function index
        tq = (o + gauss_x) / nt
        S.Bt = [
            periodic_matrix(tq, nt, d)[:, np.arange(first, first + p + 1) % nt]
            for d in range(3)
        ]  # Shape (nq, p+1) for element 0
        # Global t index of local function b on element e
        S.tidx = (np.arange(nt)[:, None] + first + np.arange(p + 1)[None, :]) % nt
        S.weights = ((gauss_w / nex)[:, None] * (gauss_w / nt)[None, :]).ravel()

        S.free = np.arange(nt, (nr - 1) * nt)  # Interior dofs of one component
        S.fixed = np.concatenate((np.arange(nt), np.arange((nr - 1) * nt, S.ndof)))
        S.free2 = np.concatenate((S.free, S.free + S.ndof))
        S.patterns = {}

        # Collocation matrices: Greville abscissae reproduce linear functions
        # in x; the centres of the periodic functions include the output
        # grid points t = j / nz
        S.greville = np.array([np.mean(S.knots[i + 1 : i + p + 1]) for i in range(nr)])
        S.centres = np.arange(nt) / nt
        S.colloc_x = clamped_matrix(S.knots, S.greville)
        S.colloc_t = periodic_matrix(S.centres, nt)

        # Boundary rows of the control net interpolate the boundaries
        S.boundary = np.empty((2, 2, nt))
        for i, (line, theta_of) in enumerate(
            ((inner, lambda v: inner_theta(v + shift)), (outer, outer_theta))
        ):
            th = theta_of(S.centres)
            pos = (line.Rvalue(th) + 1j * line.Zvalue(th) - centre) / scale
            coef = np.linalg.solve(S.colloc_t, pos)
            S.boundary[0, i] = coef.real
            S.boundary[1, i] = coef.imag
        return S

    def element_indices(S, ex):
        """Global dof of local function (a, b) on elements (ex, :)"""
        ix = ex + np.arange(p + 1)
        return (
            (ix[None, :, None] * S.nt + S.tidx[:, None, :])
            .reshape(S.nt, -1)
            .astype(np.int32)
        )

    def tensor(S, ex, dx, dt):
        """Tensor basis derivative on element row ex, shape (qx*qt, a*b)"""
        v = S.Bx[dx][ex][:, None, :, None] * S.Bt[dt][None, :, None, :]
        return v.reshape(nq * nq, -1)

    def evaluate(S, C, ex, derivs):
        """Derivatives of R and Z at the quadrature points of element row
        ex, shape (2, len(derivs), nt, qx*qt)"""
        idx = element_indices(S, ex)
        out = np.empty((2, len(derivs), S.nt, nq * nq))
        for k in range(2):
            local = C[k].ravel()[idx]  # (nt, (p+1)^2)
            for i, (dx, dt) in enumerate(derivs):
                out[k, i] = local @ tensor(S, ex, dx, dt).T
        return out

    def bilinear(Ga, w, Gb):
        """Element matrices sum_q Ga[q, a] w[e, q] Gb[q, b]"""
        return np.matmul(Ga.T[None, :, :] * w[:, None, :], Gb)

    def sparse(S, key, rows, cols, vals, rowset, colset, n):
        """CSR matrix K[rowset][:, colset] from element contributions. The
        sparsity pattern is the same for every assembly in a space, so the
        mapping of contributions to matrix entries is computed only once."""
        if key not in S.patterns:
            ndof = n * S.ndof
            rmap = np.full(ndof, -1, dtype=np.int32)
            rmap[rowset] = np.arange(len(rowset))
            cmap = np.full(ndof, -1, dtype=np.int32)
            cmap[colset] = np.arange(len(colset))
            r = rmap[np.concatenate(rows)]
            c = cmap[np.concatenate(cols)]
            mask = (r >= 0) & (c >= 0)
            keys = r[mask].astype(np.int64) * len(colset) + c[mask]
            del r, c
            entries = np.unique(keys)
            slot = np.searchsorted(entries, keys).astype(np.int32)
            del keys
            indptr = np.searchsorted(entries // len(colset), np.arange(len(rowset) + 1))
            indices = (entries % len(colset)).astype(np.int32)
            S.patterns[key] = (mask, slot, indices, indptr.astype(np.int32))
        mask, slot, indices, indptr = S.patterns[key]
        data = np.bincount(
            slot, weights=np.concatenate(vals)[mask], minlength=len(indices)
        )
        return csr_matrix((data, indices, indptr), shape=(len(rowset), len(colset)))

    def metric_A(J):
        """A = adj(G) with G = J^T J, entries (A11, A12, A22)"""
        g11 = J[0, 0] ** 2 + J[1, 0] ** 2
        g12 = J[0, 0] * J[0, 1] + J[1, 0] * J[1, 1]
        g22 = J[0, 1] ** 2 + J[1, 1] ** 2
        return g22, -g12, g11

    def initial_guess(S):
        """Linear interpolation between the boundaries"""
        g = S.greville[None, :, None]
        return (1 - g) * S.boundary[:, :1] + g * S.boundary[:, 1:]

    def make_solver():
        """Sparse direct solver which reuses an old LU factorisation as a
        GMRES preconditioner for as long as that converges quickly"""
        state = {"lu": None, "symmetric": True}

        def solve(A, b):
            A = A.tocsc()
            lu = state["lu"]
            if lu is not None and lu.shape == A.shape:
                M = LinearOperator(A.shape, lu.solve)
                x, info = gmres(A, b, M=M, rtol=1e-10, atol=0.0, restart=15, maxiter=1)
                if info == 0:
                    return x
            # The matrices are often close to symmetric positive definite, so
            # try the faster symmetric mode, unless it has failed before
            state["lu"] = lu = None  # Release memory before factorising
            accurate = False
            if state["symmetric"]:
                try:
                    lu = splu(
                        A,
                        permc_spec="MMD_AT_PLUS_A",
                        diag_pivot_thresh=0.0,
                        options={"SymmetricMode": True},
                    )
                    x = lu.solve(b)
                    accurate = np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)
                except RuntimeError:  # Zero pivot
                    pass
                state["symmetric"] = accurate
            if not accurate:
                lu = splu(A, permc_spec="MMD_AT_PLUS_A")
                x = lu.solve(b)
            state["lu"] = lu
            return x

        return solve

    def fixed_point(S, C, maxit=30, tolerance=1e-3):
        """Fixed point iteration on the nondivergence form, eqs. (22)-(24):
        A_mu(x_k) : H(x_{k+1}) = mu Laplace(x_k), tested with
        gamma * Laplace(phi). Only used to find an initial guess."""
        mu = 1e-4
        linsolve = make_solver()
        for it in range(maxit):
            rows, cols, vals = [], [], []
            rhs = np.zeros((2, S.ndof))
            for ex in range(S.nex):
                D = evaluate(S, C, ex, ((1, 0), (0, 1), (2, 0), (0, 2)))
                A11, A12, A22 = metric_A(D[:, :2])
                A11 = A11 + mu
                A22 = A22 + mu
                gamma = (A11 + A22) / (A11**2 + 2 * A12**2 + A22**2)
                w = gamma * S.weights[None, :]
                lap = tensor(S, ex, 2, 0) + tensor(S, ex, 0, 2)
                local = (
                    bilinear(lap, w * A11, tensor(S, ex, 2, 0))
                    + bilinear(lap, 2 * w * A12, tensor(S, ex, 1, 1))
                    + bilinear(lap, w * A22, tensor(S, ex, 0, 2))
                )
                lapx = D[:, 2] + D[:, 3]  # (2, nt, qx*qt)
                idx = element_indices(S, ex)
                for k in range(2):
                    np.add.at(rhs[k], idx, mu * (w * lapx[k]) @ lap)
                if "ff" not in S.patterns:
                    rows.append(np.repeat(idx, idx.shape[1], axis=1).ravel())
                    cols.append(np.tile(idx, (1, idx.shape[1])).ravel())
                vals.append(local.ravel())
            Kff = sparse(S, "ff", rows, cols, vals, S.free, S.free, 1)
            Kfb = sparse(S, "fb", rows, cols, vals, S.free, S.fixed, 1)
            change = 0.0
            for k in range(2):
                c = C[k].ravel()
                new = linsolve(Kff, rhs[k][S.free] - Kfb @ c[S.fixed])
                change = max(change, np.amax(np.abs(new - c[S.free])))
                c[S.free] = new
                C[k] = c.reshape(S.nr, S.nt)
            if verbose:
                print(f"  fixed point {it}: max change {change:.3e}")
            if change < tolerance:
                break
        return C

    def assemble(S, C, epsilon, q, need_matrix=True):
        """Residual and Jacobian of the regularised weak form, eqs. (36)-(38),
        with the diffusivity omega = det(J)^-k, q = 1 + k (section 4.3,
        method 2)"""
        res = np.zeros((2, S.ndof))
        rows, cols, vals = [], [], []
        mindet = np.inf
        for ex in range(S.nex):
            J = evaluate(S, C, ex, ((1, 0), (0, 1)))
            A11, A12, A22 = metric_A(J)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            mindet = min(mindet, np.amin(det))
            root = np.sqrt(4 * epsilon**2 + det**2)
            # Regularised det, eq. (38), avoiding cancellation for det < 0
            Rd = np.empty_like(det)
            pos = det > 0
            Rd[pos] = 0.5 * (det[pos] + root[pos])
            Rd[~pos] = 2 * epsilon**2 / (root[~pos] - det[~pos])
            dRd = Rd / root  # Derivative of Rd
            Rq = Rd**-q
            A = np.array([[A11, A12], [A12, A22]])
            F = A * Rq  # (2, 2, nt, qx*qt): F[k, beta]
            w = S.weights[None, :]
            grads = (tensor(S, ex, 1, 0), tensor(S, ex, 0, 1))
            idx = element_indices(S, ex)
            for k in range(2):
                np.add.at(res[k], idx, sum((F[k, b] * w) @ grads[b] for b in range(2)))
            if not need_matrix:
                continue
            # T[k, beta, l, m] = dF[k, beta] / dJ[l, m]
            cof = np.array([[J[1, 1], -J[1, 0]], [-J[0, 1], J[0, 0]]])
            dA = np.zeros((2, 2, 2, 2) + det.shape)
            for l in range(2):
                dA[0, 0, l, 1] = 2 * J[l, 1]
                dA[1, 1, l, 0] = 2 * J[l, 0]
                dA[0, 1, l, 0] = -J[l, 1]
                dA[0, 1, l, 1] = -J[l, 0]
                dA[1, 0, l] = dA[0, 1, l]
            T = dA * Rq - q * A[:, :, None, None] * (Rq / Rd * dRd) * cof[None, None]
            n = idx.shape[1]
            for k in range(2):
                for l in range(2):
                    local = sum(
                        bilinear(grads[b], T[k, b, l, m] * w, grads[m])
                        for b in range(2)
                        for m in range(2)
                    )
                    if "newton" not in S.patterns:
                        rows.append(np.repeat(idx + k * S.ndof, n, axis=1).ravel())
                        cols.append(np.tile(idx + l * S.ndof, (1, n)).ravel())
                    vals.append(local.ravel())
        if not need_matrix:
            return res, None, mindet
        return res, sparse(S, "newton", rows, cols, vals, S.free2, S.free2, 2), mindet

    def newton(S, C, epsilon, tolerance, q):
        """Damped Newton iteration on the regularised weak form. Converged
        when the full Newton update is smaller than the tolerance."""
        linsolve = make_solver()
        res, K, _ = assemble(S, C, epsilon, q)
        norm = np.linalg.norm(res[:, S.free])
        for it in range(maxiter):
            delta = -linsolve(K, res.ravel()[S.free2])
            delta = delta.reshape(2, -1)
            update = np.amax(np.abs(delta))
            step = 1.0
            while True:
                trial = C.reshape(2, -1).copy()
                trial[:, S.free] += step * delta
                trial = trial.reshape(C.shape)
                tres, _, tmindet = assemble(S, trial, epsilon, q, need_matrix=False)
                tnorm = np.linalg.norm(tres[:, S.free])
                if tnorm < (1 - 1e-4 * step) * norm or update < tolerance:
                    break
                step *= 0.5
                if step < 1e-4:
                    return C, False
            C = trial
            if verbose:
                print(
                    f"  newton {it} ({S.nex}x{S.nt}, eps={epsilon:.0e}, k={q - 1:g}): "
                    f"residual {tnorm:.3e}, step {step:.3f}, update {update:.3e}, "
                    f"min det {tmindet:.3e}"
                )
            if update < tolerance:
                return C, True
            res, K, _ = assemble(S, C, epsilon, q)
            norm = np.linalg.norm(res[:, S.free])
        return C, False

    def min_det(S, C):
        """Minimum of det J on a dense sample including element edges"""
        # det J is a polynomial of degree 2p-1 in each direction per element
        xs = linspace(0, 1, (2 * p + 2) * S.nex + 1)
        ts = linspace(0, 1, (2 * p + 2) * S.nt, endpoint=False)
        Cx = [clamped_matrix(S.knots, xs, d) for d in range(2)]
        Ct = [periodic_matrix(ts, S.nt, d) for d in range(2)]
        Rx, Zx = (Cx[1] @ C[k] @ Ct[0].T for k in range(2))
        Rt, Zt = (Cx[0] @ C[k] @ Ct[1].T for k in range(2))
        return np.amin(Rx * Zt - Rt * Zx)

    def weak_solve(S, C):
        """Newton on the weak form. With cell-size control, the plain
        inverse harmonic map is computed first and used as initial guess,
        as recommended in section 4.3 of the paper."""
        if homogenize:
            C, converged = newton(S, C, eps, max(tol, 1e-6), 1.0)
            if not converged:
                return C, False
        return newton(S, C, eps, tol, 1.0 + homogenize)

    def solve(S):
        """Fixed point iteration for the initial guess, then Newton on the
        weak form. Should Newton fail, retry with continuation in the
        regularisation, i.e. untangle with a softer barrier first."""
        # A rough initial guess usually suffices for Newton's method
        C0 = fixed_point(S, initial_guess(S), maxit=8, tolerance=3e-2)
        C, converged = weak_solve(S, C0.copy())
        if not converged:
            C0 = fixed_point(S, C0, maxit=30, tolerance=1e-3)
            C, converged = weak_solve(S, C0.copy())
        if not converged:
            C = C0
            for epsilon in (1e-1, 1e-2, 1e-3):
                if epsilon > eps:
                    C, _ = newton(S, C, epsilon, 1e-3, 1.0)
            C, converged = weak_solve(S, C)
        mindet = min_det(S, C)
        if verbose:
            print(f"  {S.nex}x{S.nt}: converged={converged}, min det J {mindet:.3e}")
        return C, converged, mindet

    # If the map folds, refine the spline as suggested in the paper
    for refine in range(max_refine + 1):
        S = make_space(2**refine * resolution * (nx - 1), 2**refine * resolution * nz)
        C, converged, mindet = solve(S)
        if mindet > 0:
            if not converged:
                warnings.warn(
                    "grid_inverse_harmonic: Newton iteration did not converge; the "
                    "grid is valid but not the exact solution of the PDE"
                )
            break
        if refine < max_refine:
            warnings.warn(
                f"grid_inverse_harmonic: map folds with {S.nex}x{S.nt} "
                "spline elements, refining"
            )
    else:
        raise RuntimeError(
            "grid_inverse_harmonic: failed to find a grid without folds "
            f"(min det J = {mindet:.3e}). Try more points or a smaller homogenize."
        )
    knots, nt = S.knots, S.nt

    # Evaluate the spline map on the output grid. Extra points outside the
    # boundaries continue the map with its second order Taylor expansion at
    # the boundary, so that the metric stays continuously differentiable.
    dx = 1.0 / (nx - 1)
    xvals = np.arange(-nx_inner, nx + nx_outer) * dx
    xvals[nx_inner + nx - 1] = 1.0  # Make sure the outer boundary is exact
    xclip = np.clip(xvals, 0.0, 1.0)
    delta = (xvals - xclip)[:, None]
    Cx = sum(
        delta**d / (1, 1, 2)[d] * clamped_matrix(knots, xclip, d) for d in range(3)
    )
    Ct = periodic_matrix(np.arange(nz) / nz, nt)
    R = Cx @ C[0] @ Ct.T * scale + centre.real
    Z = Cx @ C[1] @ Ct.T * scale + centre.imag

    # The spline map is valid, but the quadrilaterals between the grid
    # points can still be non-convex where the grid is strongly sheared,
    # and extrapolated cells may fold
    Rp = np.concatenate((R, R[:, :1]), axis=1)
    Zp = np.concatenate((Z, Z[:, :1]), axis=1)
    corners = [(Rp[:-1, :-1], Zp[:-1, :-1]), (Rp[1:, :-1], Zp[1:, :-1])]
    corners += [(Rp[1:, 1:], Zp[1:, 1:]), (Rp[:-1, 1:], Zp[:-1, 1:])]
    cross = np.minimum.reduce(
        [
            (r2 - r1) * (z0 - z1) - (z2 - z1) * (r0 - r1)
            for (r0, z0), (r1, z1), (r2, z2) in (
                (corners[i - 1], corners[i], corners[(i + 1) % 4]) for i in range(4)
            )
        ]
    )
    for rows, where in (
        (slice(0, nx_inner), "extrapolated inside the inner boundary"),
        (slice(nx_inner, nx_inner + nx - 1), "between the boundaries"),
        (slice(nx_inner + nx - 1, None), "extrapolated outside the outer boundary"),
    ):
        bad = np.sum(cross[rows] <= 0)
        if bad:
            warnings.warn(
                f"grid_inverse_harmonic: {bad} cells {where} are folded or not convex"
            )

    if show and plotting_available:
        plt.figure()
        r, z = inner.position(linspace(0, 2 * pi, 10 * nz))
        plt.plot(r, z, "k")
        r, z = outer.position(linspace(0, 2 * pi, 10 * nz))
        plt.plot(r, z, "k")
        plt.plot(R, Z, "b", lw=0.5)
        plt.plot(np.transpose(R), np.transpose(Z), "b", lw=0.5)
        plt.gca().set_aspect("equal")
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
