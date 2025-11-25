import itertools
from multiprocessing import Pool

import numpy as np
from scipy.integrate import odeint

try:
    import eudist

    has_eudist = True
except ImportError:
    has_eudist = False

from .field import _set_config


class FieldTracer(object):
    """A class for following magnetic field lines

    Parameters
    ----------
    field : :py:obj:`~zoidberg.field.MagneticField`
        A Zoidberg MagneticField instance

    """

    def __init__(self, field):
        self.field_direction = field.field_direction

    def follow_field_lines(self, x_values, z_values, y_values, rtol=None):
        """Uses field_direction to follow the magnetic field
        from every grid (x,z) point at toroidal angle y
        through a change in toroidal angle dy

        Parameters
        ----------
        x_values : array_like
            Starting x coordinates
        z_values : array_like
            Starting z coordinates
        y_values : array_like
            y coordinates to follow the field line to. y_values[0] is
            the starting position
        rtol : float, optional
            The relative tolerance to use for the integrator. If None,
            use the default value

        Returns
        -------
        result : numpy.ndarray
            Field line ending coordinates

            The first dimension is y, the last is (x,z). The
            middle dimensions are the same shape as [x|z]:
            [0,...] is the initial position
            [...,0] are the x-values
            [...,1] are the z-values
            If x_values is a scalar and z_values a 1D array, then result
            has the shape [len(y), len(z), 2], and vice-versa.
            If x_values and z_values are 1D arrays, then result has the shape
            [len(y), len(x), 2].
            If x_values and z_values are 2D arrays, then result has the shape
            [len(y), x.shape[0], x.shape[1], 2].

        """

        # Ensure all inputs are NumPy arrays
        x_values = np.atleast_1d(x_values)
        y_values = np.atleast_1d(y_values)
        z_values = np.atleast_1d(z_values)

        if len(y_values) < 2:
            raise ValueError("There must be at least two elements in y_values")
        if len(y_values.shape) > 1:
            raise ValueError("y_values must be 1D")

        if x_values.shape != z_values.shape:
            # Make the scalar the same shape as the array
            if x_values.size == 1:
                x_values = np.zeros(z_values.shape) + x_values
            elif z_values.size == 1:
                z_values = np.zeros(x_values.shape) + z_values
            else:
                raise ValueError(
                    "x_values and z_values must be the same size, or one must be a scalar"
                )

        array_shape = x_values.shape

        # Position vector must be 1D - so flatten before passing to
        # integrator, then reshape after
        if len(x_values.shape) > 1:
            x_values = x_values.flatten()
            z_values = z_values.flatten()

        # If too many points are calculated at once (>~ 1e4) then
        # memory errors occur.

        if len(x_values) < 1000:
            position = np.column_stack((x_values, z_values)).flatten()

            result = odeint(
                self.field_direction, position, y_values, args=(True,), rtol=rtol
            )
        else:
            # Split into smaller pieces

            nchunks = int(len(x_values) / 1000)

            # Splitting x and z values separately to avoid ending up with
            # an odd number of points in position arrays
            x_values = np.array_split(x_values, nchunks)
            z_values = np.array_split(z_values, nchunks)

            # Combine x,z into flattened position arrays
            chunks = [
                np.column_stack((x, z)).flatten() for x, z in zip(x_values, z_values)
            ]

            # Process in chunks. Note: multiprocessing has trouble with closures
            # so fails in Python 2. Python 3 may work
            results = [
                odeint(self.field_direction, chunk, y_values, args=(True,), rtol=rtol)
                for chunk in chunks
            ]

            # Concatenate results into a single array
            result = np.concatenate(results, axis=1)

        return result.reshape(y_values.shape + array_shape + (2,))


class FieldTracerReversible(object):
    """Traces magnetic field lines in a reversible way by using
    trapezoidal integration:

    .. math::

       pos_{n+1} = pos_n + 0.5*( f(pos_n) + f(pos_{n+1}) )*dy

    This requires a Newton iteration to solve the nonlinear set of equations
    for the unknown ``pos_{n+1}``.

    Parameters
    ----------
    field : :py:obj:`~zoidberg.field.MagneticField`
        A Zoidberg MagneticField instance
    rtol : float, optional
        Tolerance applied to changes in dx**2 + dz**2
    eps : float, optional
        Change in x,z used to calculate finite differences of magnetic
        field direction
    nsteps : int, optional
        Number of sub-steps between outputs

    """

    def __init__(self, field, rtol=1e-8, eps=1e-5, nsteps=20):
        self.field_direction = field.field_direction
        self.rtol = float(rtol)
        self.eps = float(eps)
        self.nsteps = int(nsteps)

    def follow_field_lines(
        self, x_values, z_values, y_values, rtol=None, eps=None, nsteps=None
    ):
        """Uses field_direction to follow the magnetic field
        from every grid (x,z) point at toroidal angle y
        through a change in toroidal angle dy

        Parameters
        ----------
        x_values : array_like
            Starting x coordinates
        z_values : array_like
            Starting z coordinates
        y_values : array_like
            y coordinates to follow the field line to. y_values[0] is
            the starting position
        rtol : float, optional
            Tolerance applied to changes in dx**2 + dz**2. If None,
            use the default value
        eps : float, optional
            Change in x,z used to calculate finite differences of magnetic
            field direction
        nsteps : int, optional
            Number of sub-steps between outputs

        Returns
        -------
        result : numpy.ndarray
            Field line ending coordinates

            The first dimension is y, the last is (x,z). The
            middle dimensions are the same shape as [x|z]:
            [0,...] is the initial position
            [...,0] are the x-values
            [...,1] are the z-values
            If x_values is a scalar and z_values a 1D array, then result
            has the shape [len(y), len(z), 2], and vice-versa.
            If x_values and z_values are 1D arrays, then result has the shape
            [len(y), len(x), 2].
            If x_values and z_values are 2D arrays, then result has the shape
            [len(y), x.shape[0], x.shape[1], 2].

        """

        # Check settings, use defaults if not given
        if rtol is None:
            rtol = self.rtol  # Value set in __init__
        rtol = float(rtol)
        if eps is None:
            eps = self.eps
        eps = float(eps)
        if nsteps is None:
            nsteps = self.nsteps
        nsteps = int(nsteps)

        # Ensure all inputs are NumPy arrays
        x_values = np.atleast_1d(x_values)
        y_values = np.atleast_1d(y_values)
        z_values = np.atleast_1d(z_values)

        if len(y_values) < 2:
            raise ValueError("There must be at least two elements in y_values")
        if len(y_values.shape) > 1:
            raise ValueError("y_values must be 1D")

        if x_values.shape != z_values.shape:
            # Make the scalar the same shape as the array
            if x_values.size == 1:
                x_values = np.zeros(z_values.shape) + x_values
            elif z_values.size == 1:
                z_values = np.zeros(x_values.shape) + z_values
            else:
                raise ValueError(
                    "x_values and z_values must be the same size, or one must be a scalar"
                )

        array_shape = x_values.shape

        result = np.zeros((len(y_values),) + array_shape + (2,))

        # Starting position
        x_pos = x_values
        z_pos = z_values
        y_pos = y_values[0]

        result[0, ..., 0] = x_pos
        result[0, ..., 1] = z_pos

        for yindex, y_next in enumerate(y_values[1:]):
            yindex += 1  # Since we chopped off the first y_value

            # Split into sub-steps
            dy = (y_next - y_pos) / float(nsteps)
            for step in range(nsteps):
                # Evaluate gradient at current position
                dxdy, dzdy = self.field_direction((x_pos, z_pos), y_pos)

                # Half-step values
                x_half = x_pos + 0.5 * dxdy * dy
                z_half = z_pos + 0.5 * dzdy * dy

                # Now need to find the roots of a nonlinear equation
                #
                # f = 0.5*(dpos/dy)(pos_next)*dy - (pos_next - pos_half) = 0
                #

                # Use Euler method to get starting guess
                x_pos += dxdy * dy
                z_pos += dzdy * dy
                y_pos += dy

                while True:
                    dxdy, dzdy = self.field_direction((x_pos, z_pos), y_pos)

                    # Calculate derivatives (Jacobian matrix) by finite difference
                    dxdy_xe, dzdy_xe = self.field_direction((x_pos + eps, z_pos), y_pos)
                    dxdy_x = (dxdy_xe - dxdy) / eps
                    dzdy_x = (dzdy_xe - dzdy) / eps

                    dxdy_ze, dzdy_ze = self.field_direction((x_pos, z_pos + eps), y_pos)
                    dxdy_z = (dxdy_ze - dxdy) / eps
                    dzdy_z = (dzdy_ze - dzdy) / eps

                    # The function we are trying to find the roots of:
                    fx = 0.5 * dxdy * dy - x_pos + x_half
                    fz = 0.5 * dzdy * dy - z_pos + z_half

                    # Now have a linear system to solve
                    #
                    # (x_pos)  -= ( dfx/dx   dfx/dz )^-1 (fx)
                    # (z_pos)     ( dfz/dx   dfz/dz )    (fz)

                    dfxdx = 0.5 * dxdy_x * dy - 1.0
                    dfxdz = 0.5 * dxdy_z * dy
                    dfzdx = 0.5 * dzdy_x * dy
                    dfzdz = 0.5 * dzdy_z * dy - 1.0

                    determinant = dfxdx * dfzdz - dfxdz * dfzdx
                    # Note: If determinant is too small then dt should be reduced

                    dx = (dfzdz * fx - dfxdz * fz) / determinant
                    dz = (dfxdx * fz - dfzdx * fx) / determinant

                    x_pos -= dx
                    z_pos -= dz
                    # Check for convergence within tolerance
                    if np.amax(dx**2 + dz**2) < rtol:
                        break
                # Finished Newton iteration, taken step to (x_pos,y_pos,z_pos)
            # Finished sub-steps, reached y_pos = y_next
            result[yindex, ..., 0] = x_pos
            result[yindex, ..., 1] = z_pos

        return result


def trace_poincare(
    magnetic_field, xpos, zpos, yperiod, nplot=3, y_slices=None, revs=20, nover=20
):
    """Trace a Poincare graph of the field lines

    Does no plotting, see :py:func:`zoidberg.plot.plot_poincare`

    Parameters
    ----------
    magnetic_field : :py:obj:`~zoidberg.field.MagneticField`
        Magnetic field object
    xpos, zpos : array_like
        Starting X, Z locations
    yperiod : float
        Length of period in y domain
    nplot : int, optional
        Number of equally spaced y-slices to trace to
    y_slices : list of ints
        List of y-slices to plot; overrides `nplot`
    revs : int, optional
        Number of revolutions (times around y)
    nover : int, optional
        Over-sample. Produced additional points in y then discards.
        This seems to be needed for accurate results in some cases

    Returns
    -------
    coords, y_slices
        coords is a Numpy array of data::

            [revs, nplot, ..., R/Z]

        where the first index is the revolution, second is the y slice,
        and last is 0 for R, 1 for Z. The middle indices are the shape
        of the input xpos,zpos

    """

    if nplot is None and y_slices is None:
        raise ValueError("nplot and y_slices cannot both be None")

    if y_slices is not None:
        y_slices = np.asarray(y_slices)

        if np.amin(y_slices) < 0.0 or np.amax(y_slices) > yperiod:
            raise ValueError(
                "y_slices must all be between 0.0 and yperiod ({yperiod})".format(
                    yperiod=yperiod
                )
            )
        # Make sure y_slices is monotonically increasing
        y_slices.sort()
        # If y_slices is given, then nplot is the number of slices
        nplot = len(y_slices)
    else:
        # nplot equally spaced phi slices
        nplot = int(nplot)
        y_slices = np.linspace(0, yperiod, nplot, endpoint=False)

    # Extend the domain from [0,yperiod] to [0,revs*yperiod]

    revs = int(revs)
    y_values = y_slices[:]
    for n in np.arange(1, revs):
        y_values = np.append(y_values, n * yperiod + y_values[:nplot])

    nover = int(nover)  # Over-sample
    y_values_over = np.zeros((nplot * revs * nover - (nover - 1)))
    y_values_over[::nover] = y_values
    for i in range(1, nover):
        y_values_over[i::nover] = (float(i) / float(nover)) * y_values[1:] + (
            float(nover - i) / float(nover)
        ) * y_values[:-1]

    # Starting location
    xpos = np.asarray(xpos)
    zpos = np.asarray(zpos)

    field_tracer = FieldTracer(magnetic_field)
    result = field_tracer.follow_field_lines(xpos, zpos, y_values_over)

    result = result[::nover, ...]  # Remove unneeded points

    # Reshape data. Loops fastest over planes (nplot)
    result = np.reshape(result, (revs, nplot) + result.shape[1:])

    return result, y_slices


class CachingFieldTracer:
    def __init__(
        self, tracer, cachename="_zoidberg_cache_tracing_%10.10s", debug=False
    ):
        self.tracer = tracer
        self.cachename = cachename
        self.debug = debug

    def _myhash(self, *args):
        """

        Compute a digest over some numerical data

        """
        import hashlib

        m = hashlib.md5()
        for arg in args:
            s = "-".join([str(x) for x in np.array([arg]).ravel()]).encode()
            if self.debug:
                print(s)
            m.update(s)
        return m.hexdigest()

    def _guess_shape(self, x_values, y_values):
        return len(y_values), *x_values.shape, 2

    def follow_field_lines(self, x_values, z_values, y_values, **kwargs):
        fn = self.cachename % self._myhash(x_values, z_values, y_values)
        try:
            dat = np.loadtxt(fn)
            dat.shape = self._guess_shape(x_values, y_values)
            return dat
        except FileNotFoundError:
            if self.debug:
                print("no cache", fn)
        ret = self.tracer.follow_field_lines(x_values, z_values, y_values, **kwargs)
        shape = ret.shape
        if self.debug:
            print(f"saving {ret.shape}, {x_values.shape}")
        ret.shape = -1, shape[-1]
        np.savetxt(fn, ret)
        assert shape == self._guess_shape(
            x_values, y_values
        ), f"Estimated {self._guess_shape(x_values, y_values)} but got {shape}"
        ret.shape = shape
        return ret


def _get_value(*args):
    for k in args:
        if k is not None:
            return k
    raise ValueError(f"Expected at least one non-None value in {args}")


class FieldTracerWeb:
    """A class for following magnetic field lines

    Parameters
    ----------
    field : :py:obj:`~zoidberg.field.MagneticField`
        A Zoidberg MagneticField instance

    """

    def __init__(
        self,
        config=None,
        configId=None,
        timeout=1,
        chunk=10000,
        stepsize=None,
        retry=None,
        run_timeout=None,
    ):
        try:
            import requests
        except ModuleNotFoundError:
            requests = None
        # First check whether the webservice is available, as the default timeout is rather slow.
        self.url = "http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl"
        if requests:
            requests.get(self.url, timeout=timeout).raise_for_status()

        self.timeout = run_timeout
        self.retry = retry
        self.config = config
        self.configId = configId
        self.stepsize = stepsize
        if self.config:
            assert self.configId is None
        else:
            assert self.configId is not None

        # Check tracing direction
        pos = self.flt.types.Points3D()
        pos.x1 = [6]
        pos.x2 = [0]
        pos.x3 = [0]

        lineTask = self.flt.types.LineTracing()
        lineTask.numSteps = 1

        task = self.flt.types.Task()
        task.step = 0.001
        task.lines = lineTask

        line = self.flt.service.trace(pos, self.getConfig(), task, None, None).lines[0]
        phi = np.arctan2(line.vertices.x2, line.vertices.x1)[-1]
        assert phi != 0
        self.isforward = phi > 0
        self.chunk = chunk

    def follow_field_lines(
        self,
        x_values,
        z_values,
        y_values,
        rtol=None,
        stepsize=None,
        timeout=None,
        retry=None,
        chunk=None,
    ):
        """Uses field_direction to follow the magnetic field
        from every grid (x,z) point at toroidal angle y
        through a change in toroidal angle dy

        Parameters
        ----------
        x_values : array_like
            Starting x coordinates
        z_values : array_like
            Starting z coordinates
        y_values : array_like
            y coordinates to follow the field line to. y_values[0] is
            the starting position
        rtol : float, optional
            Currently ignored. Use stepsize instead
        stepsize : float, optional
            The stepsize for the integrator. Sets accuracy and speed.
        timeout : float, optional
            How long to wait. Sometimes the server does not return and
            the client keeps waiting for ever, otherwise.
        retry : int, optional
            How often to retry failed calculations
        chunk: int or None, optional
            Set chunking size, overwrite default if not None

        Returns
        -------
        result : numpy.ndarray
            Field line ending coordinates

            The first dimension is y, the last is (x,z). The
            middle dimensions are the same shape as [x|z]:
            [0,...] is the initial position
            [...,0] are the x-values
            [...,1] are the z-values
            If x_values is a scalar and z_values a 1D array, then result
            has the shape [len(y), len(z), 2], and vice-versa.
            If x_values and z_values are 1D arrays, then result has the shape
            [len(y), len(x), 2].
            If x_values and z_values are 2D arrays, then result has the shape
            [len(y), x.shape[0], x.shape[1], 2].

        """
        retry = _get_value(retry, self.retry, 3)
        timeout = _get_value(timeout, self.timeout, 1800)

        # Ensure all inputs are NumPy arrays
        x_values = np.atleast_1d(x_values)
        y_values = np.atleast_1d(y_values)
        z_values = np.atleast_1d(z_values)

        if len(y_values) < 2:
            raise ValueError("There must be at least two elements in y_values")
        if len(y_values.shape) > 1:
            raise ValueError("y_values must be 1D")

        if x_values.shape != z_values.shape:
            # Make the scalar the same shape as the array
            if x_values.size == 1:
                x_values = np.zeros(z_values.shape) + x_values
            elif z_values.size == 1:
                z_values = np.zeros(x_values.shape) + z_values
            else:
                raise ValueError(
                    "x_values and z_values must be the same size, or one must be a scalar"
                )

        array_shape = x_values.shape

        # Position vector must be 1D - so flatten before passing to
        # integrator, then reshape after
        if len(x_values.shape) > 1:
            x_values = x_values.flatten()
            z_values = z_values.flatten()
        chunk = chunk or self.chunk
        if self.config is None and x_values.size > chunk:
            with Pool(12) as pool:

                def start(i):
                    return pool.apply_async(
                        self._follow_field_lines,
                        (
                            x_values[i : i + chunk],
                            z_values[i : i + chunk],
                            y_values,
                            rtol,
                            stepsize,
                        ),
                    )

                # Start parallel calculation
                results = [start(i) for i in range(0, len(x_values), chunk)]
                # Wait for result and combine
                for j, i in enumerate(range(0, len(x_values), chunk)):
                    while retry > 0:
                        try:
                            results[j] = results[j].get(timeout=timeout)
                            break
                        except Exception as e:
                            retry -= 1
                            if not retry > 0:
                                raise
                            print(
                                f"Fieldlinetracer failed {e} - going to retry {retry} times"
                            )
                            results[j] = start(i)
                try:
                    results = np.concatenate(results, axis=1)
                except ValueError:
                    print(results)
                    raise
                    raise ValueError(
                        f"np.concatenate failed, shapes where {', '.join([str(x.shape) for x in results])}."
                    )
        else:
            results = [
                self._follow_field_lines(
                    x_values[i : i + chunk],
                    z_values[i : i + chunk],
                    y_values,
                    rtol,
                    stepsize,
                )
                for i in range(0, len(x_values), chunk)
            ]
            # Wait for result and combine
            results = np.concatenate(results, axis=1)

        return results.reshape(y_values.shape + array_shape + (2,))

    def getConfig(self):
        if self.config is not None:
            self.config.inverseField = False
            return self.config
        config = self.flt.types.MagneticConfig()
        config = _set_config(config, self.configId)
        config.inverseField = False
        return config

    @property
    def flt(self):
        from osa import Client

        return Client(self.url)

    def _follow_field_lines(self, x_values, z_values, y_values, rtol, stepsize):
        from time import sleep

        sleep(np.random.random(1)[0])
        p = self.flt.types.Points3D()
        p.x1 = x_values * np.cos(y_values[0])
        p.x2 = x_values * np.sin(y_values[0])
        p.x3 = z_values

        task = self.flt.types.Task()
        task.step = stepsize or self.stepsize or 0.01
        task.linesPhi = self.flt.types.LinePhiSpan()

        def xor(x, y):
            return bool((x and not y) or (not x and y))

        dphi = y_values[1] - y_values[0]
        config = self.getConfig()
        config.inverseField = xor(dphi > 0, self.isforward)
        result = np.empty((len(y_values), len(p.x1), 2))
        result[0, :, 0] = x_values
        result[0, :, 1] = z_values
        phi0 = y_values[0]
        for i, phi in enumerate(y_values):
            if i == 0:
                continue
            task.linesPhi.phi = phi - phi0
            res = self.flt.service.trace(p, config, task)
            for j, curve in enumerate(res.lines):
                ps = curve.vertices
                result[i, j, 0] = np.sqrt(ps.x2[-1] ** 2 + ps.x1[-1] ** 2)
                result[i, j, 1] = ps.x3[-1]
        return result


if has_eudist:

    class _PolyMesh(eudist.PolyMesh):
        def __init__(self, x, y):
            super().__init__()
            self.r = x
            self.z = y
            self.grid = np.array([x, y]).transpose(1, 2, 0)

else:

    class _PolyMesh:
        def __init__(self, x, y):
            raise RuntimeError(
                "eudist is needed to use PolyMesh. Maybe use pip install eudist?"
            )


class EMC3FieldTracer(FieldTracer):
    """A class for following magnetic field lines provided by an EMC3 grid.

    Parameters
    ----------
    field : :py:obj:`~zoidberg.field.MagneticField`
        A Zoidberg MagneticField instance

    """

    def __init__(self, field):
        import xarray as xr

        self.field_direction = field.field_direction
        self.field = field
        self.ds = field.ds
        self.ds_min = xr.Dataset(
            dict(
                R_bounds=self.ds["R_bounds"],
                z_bounds=self.ds["z_bounds"],
                phi_bounds=self.ds["phi_bounds"],
            )
        )
        for k in self.ds:
            if k.endswith("_dims"):
                self.ds_min[k] = self.ds[k]
        first = [self.makeMeshInd(0, zid) for zid in self._iterZones()]
        last = [self.makeMeshInd(-1, zid) for zid in self._iterZones()]
        self.firstlast = (first, last)

    def follow_field_lines(self, x_values, z_values, y_values, rtol=None):
        meshes = [self.makeMeshes(phi) for phi in y_values]
        assert x_values.shape == z_values.shape
        out = np.empty((len(y_values), *x_values.shape, 2))
        ij = -1
        zid = 0
        # print([range(x) for x in x_values.shape])
        for i in itertools.product(*[range(x) for x in x_values.shape]):
            rz = np.array((x_values[i], z_values[i]))
            ab, ij, zid = self.rz_to_ab(rz, meshes[0], ij, zid)
            if x_values.shape == ():
                i = None
            out[(0, *i)] = rz
            ij1, zid1 = ij, zid
            cache = {0: (ab, ij, zid)}
            for j in range(1, len(meshes)):
                perBC = meshes[j][zid].perBC - meshes[0][zid].perBC
                if perBC not in cache:
                    first, last = (0, 1) if perBC > 0 else (1, 0)
                    for _ in range(abs(perBC)):
                        rz1 = self._ab_to_rz(ab, self.firstlast[last][zid1].grid, ij1)
                        ab, ij1, zid1 = self.rz_to_ab(
                            rz1, self.firstlast[first], ij1, zid1
                        )
                    cache[perBC] = self.rz_to_ab(rz, meshes[0], ij, zid)
                ab, ij1, zid1 = cache[perBC]
                out[(j, *i)] = self._ab_to_rz(ab, meshes[j][zid1].grid, ij1)
        return out

    def _rz_to_ab(self, rz, grid, ij):
        _, nz, _ = grid.shape
        nz -= 1
        i, j = ij // nz, ij % nz
        ABCD = grid[i : i + 2, j : j + 2]
        if ABCD.shape != (2, 2, 2):
            print(grid.shape)
            print(rz)
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(grid[:, :, 0], grid[:, :, 1])
            plt.plot(grid[:, :, 0].T, grid[:, :, 1].T)
            plt.plot(*rz, "ro")
            plt.show()
            pass
        assert ABCD.shape == (
            2,
            2,
            2,
        ), f"{ABCD.shape} {ij} = {i} * {nz} + {j}, {grid.shape}"
        A = ABCD[0, 0]
        a = ABCD[0, 1] - A
        b = ABCD[1, 0] - A
        c = ABCD[1, 1] - A - a - b
        rz0 = rz - A

        def fun(albe):
            al, be = albe
            return rz0 - a * al - b * be - c * al * be

        def J(albe):
            al, be = albe
            return np.array([-a - c * be, -b - c * al])

        tol = 1e-13
        albe = np.ones(2) / 2
        # while True:
        for i in range(100):
            assert np.all(np.isfinite(albe))
            albe = albe - np.linalg.inv(J(albe).T) @ fun(albe)
            res = np.sum(fun(albe) ** 2)
            if res < tol:
                return albe
        import scipy.optimize as sopt

        def fun1(x0, a, b, c, rz0):
            al, be = x0
            return al * a + be * b + al * be * c - rz0

        res = sopt.least_squares(fun1, np.zeros(2), args=(a, b, c, rz0))
        return res.x

    def _ab_to_rz(self, ab, grid, ij):
        _, nz, _ = grid.shape
        nz -= 1
        i, j = ij // nz, ij % nz
        A = grid[i, j]
        a = grid[i, j + 1] - A
        b = grid[i + 1, j] - A
        c = grid[i + 1, j + 1] - A - a - b
        al, be = ab
        return A + al * a + be * b + al * be * c

    def rz_to_ab(self, rz, meshes, ij, zid):
        for mesh in itertools.chain(meshes[zid:], meshes[:zid]):
            ij = mesh.find_cell(rz, ij)
            if ij >= 0:
                return self._rz_to_ab(rz, mesh.grid, ij), ij, mesh.zid

        for mind in 1e-3, 1e-2, 1:
            mind = mind**2
            for mesh in itertools.chain(meshes[zid:], meshes[:zid]):
                l2 = (rz[0] - mesh.r) ** 2 + (rz[1] - mesh.z) ** 2
                ij = np.argmin(l2)
                if l2.flat[ij] < mind:
                    nr, nz = mesh.r.shape
                    if ij >= (nr - 1) * nz:
                        ij -= nz
                    return self._rz_to_ab(rz, mesh.grid, ij), ij, mesh.zid

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        plt.figure()
        for mesh, clr in zip(meshes, itertools.cycle(mcolors.TABLEAU_COLORS.values())):
            plt.plot(mesh.r.T, mesh.z.T, c=clr)
            plt.plot(mesh.r, mesh.z, c=clr)
        plt.plot(*rz, "ro")
        plt.title("We left the domain!")
        plt.show()
        raise RuntimeError("We left the domain")

    def _iterZones(self):
        if "zone" in self.ds.dims:
            return range(len(self.ds.zone))
        return range(1)

    def makeMeshes(self, phi):
        return [self.makeMesh(phi, zid) for zid in self._iterZones()]

    def makeMeshInd(self, phi, zid):
        di = self.ds_min.emc3.isel(zone=zid)
        if phi == -1:
            di = di.isel(phi=phi)
            di = di.isel(delta_phi=phi)
        else:
            di = di.emc3.isel(phi=phi)
        Rsrc = di.emc3["R_corners"]
        Zsrc = di.emc3["z_corners"]

        mesh = _PolyMesh(Rsrc.data, Zsrc.data)
        mesh.zid = zid
        return mesh

    def makeMesh(self, phi, zid):
        di = self.ds_min.emc3.isel(zone=zid)
        pmin = np.min(di["phi_bounds"])
        pmax = np.max(di["phi_bounds"])
        perBC = 0
        if not (pmin < phi < pmax):
            deltap = pmax - pmin
            while phi < pmin:
                phi += deltap
                perBC += 1
            while phi > pmax:
                phi -= deltap
                perBC -= 1
        di = di.emc3.sel(phi=phi)
        Rsrc = di.emc3["R_corners"]
        Zsrc = di.emc3["z_corners"]

        mesh = _PolyMesh(Rsrc.data, Zsrc.data)
        mesh.zid = zid
        mesh.phi = phi
        mesh.perBC = perBC
        return mesh
