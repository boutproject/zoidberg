import numpy as np


def c2(f, axis, periodic=False):
    if periodic:
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis)) / 2
    sl1 = [slice(None) for _ in f.shape]
    sl2 = [slice(None) for _ in f.shape]
    out = np.empty_like(f)

    t = tuple

    sl2[axis] = slice(1, -1)
    sl1[axis] = slice(2, None)
    out[t(sl2)] = f[t(sl1)]
    sl1[axis] = slice(None, -2)
    out[t(sl2)] -= f[t(sl1)]

    sl2[axis] = 0
    sl1[axis] = 2
    out[t(sl2)] = f[t(sl1)]
    out[t(sl2)] -= f[t(sl2)]

    sl2[axis] = -1
    sl1[axis] = -3
    out[t(sl2)] = f[t(sl2)]
    out[t(sl2)] -= f[t(sl1)]

    out /= 2
    return out


def c4(f, axis, periodic=False):
    if periodic:

        def fi(i):
            return np.roll(f, -i, axis=axis)

        return (-fi(2) + 8 * fi(1) - 8 * fi(-1) + fi(-2)) / 12

    def fi(i):
        slc0 = list(slc)
        if isinstance(slc0[axis], int):
            slc0[axis] += i
        else:
            sli = slc0[axis]
            start = i if sli.start is None else sli.start + i
            stop = -1 + i if sli.stop is None else sli.stop + i
            if sli.stop < 0 and stop == 0:
                stop = None
            slc0[axis] = slice(start, stop)
        return f[tuple(slc0)]

    t = tuple

    out = np.empty_like(f)
    slc = [slice(None) for _ in f.shape]

    slc[axis] = slice(2, -2)
    # out[t(slc)] = 0.083333333333333333 * fi(-2) + -0.66666666666666666 * fi(-1) + 0.66666666666666666 * fi(1) + -0.08333333333333333 * fi(2)
    out[t(slc)] = (-fi(2) + 8 * fi(1) - 8 * fi(-1) + fi(-2)) / 12

    slc[axis] = 0
    out[t(slc)] = (
        -2.0833333333333333 * fi(0)
        + 4 * fi(1)
        - 3 * fi(2)
        + 1.3333333333333333 * fi(3)
        - 0.25 * fi(4)
    )

    slc[axis] = 1
    out[t(slc)] = (
        -0.25 * fi(-1)
        - 0.83333333333333333 * fi(0)
        + 1.5 * fi(1)
        - 0.5 * fi(2)
        + 0.083333333333333333 * fi(3)
    )

    slc[axis] = -2
    out[t(slc)] = (
        -0.083333333333333333 * fi(-3)
        + 0.5 * fi(-2)
        - 1.5 * fi(-1)
        + 0.83333333333333333 * fi(0)
        + 0.25 * fi(1)
    )

    slc[axis] = -1
    out[t(slc)] = (
        0.25 * fi(-4)
        - 1.3333333333333333 * fi(-3)
        + 3 * fi(-2)
        - 4 * fi(-1)
        + 2.08333333333333333 * fi(0)
    )
    return out


def get_dist(RZ_coords, y_coords, refine=100):
    """
    This function takes the trace from a field line tracer and calculate
    the length along the points.

    This is done by interpolating the points in cylindircal coordinates using
    a spline. Then the line is transformed to cartesian coordinates, where the
    sum of the point wise distance is computed.
    """
    from scipy.interpolate import CubicSpline as interp

    assert RZ_coords.shape[-1] == 2
    RZ_coords = RZ_coords[..., 0], RZ_coords[..., 1]

    y_inter = interp(np.linspace(0, 1, len(y_coords)), y_coords)

    if 1:
        y_fine = y_inter(np.linspace(0, 1, (len(y_coords) - 1) * refine + 1))

        R_fine, Z_fine = [interp(y_coords, x)(y_fine) for x in RZ_coords]
    else:
        y_fine = y_coords
        R_fine, Z_fine = RZ_coords

    slicer = [None for _ in R_fine.shape]
    slicer[0] = slice(None, None)
    y_fine = y_fine[tuple(slicer)]

    X_fine = np.cos(y_fine) * R_fine
    Y_fine = np.sin(y_fine) * R_fine

    dXYZs = [(x[1:] - x[:-1]) ** 2 for x in [X_fine, Y_fine, Z_fine]]
    ds2 = np.sum(dXYZs, axis=0)
    ds = np.sqrt(ds2)
    s = np.sum(ds, axis=0)

    return s
