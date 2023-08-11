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
