#!/usr/bin/env python
# coding: utf-8

import time


import sys

import numpy as np
from boututils.datafile import DataFile as DF

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

time_start = time.time()  # noqa
verbose = 1


def toCent(RZ):
    RZc = RZ + np.roll(RZ, 1, axis=-1)
    RZc = RZc[:, 1:] + RZc[:, :-1]
    return RZc / 4


def l2(x):
    return np.sqrt(np.sum(x**2, axis=0))


org_print = print


def my_print(*args):
    args = f"{time.time() - time_start:10.3f} s: ", *args
    org_print(*args)


def log(*args):
    if verbose > 1:
        my_print(*args)


def print(*args):
    if verbose:
        my_print(*args)


def load(fn):
    with DF(fn) as f:
        RZ = [f[k] for k in "RZ"]
        log("opened")
        RZ = np.array(RZ)
        _, a, b, c = RZ.shape

        coefsX = np.zeros((a - 1, b, c, 2))
        coefsZ = np.zeros((a - 2, b, c, 2))
        for d, x1 in zip((coefsX, coefsZ), "XZ"):
            for i, x2 in enumerate("XZ"):
                key = f"dagp_fv_{x1}{x2}"
                fixup2(d[..., i], f[key])
        key = "dagp_fv_volume"
        volume = f[key]
    log("done reading")
    return RZ, volume, coefsX, coefsZ


def doit(pols, plot=False):
    RZs = []

    ### Calculate Volume of the cell
    #
    ### Go in a line around the cell

    A = np.empty((pols[0].nx, len(pols), pols[0].nz))
    Ar = np.empty_like(A)
    todo = enumerate(pols)
    if tqdm:
        todo = tqdm(todo, total=len(pols))

    for gi, g in todo:
        n = 12
        x0 = np.arange(g.nx)[1:-1, None, None]
        y0 = np.arange(g.nz)[None, :, None]
        x1 = x0[:, :, 0]
        y1 = y0[:, :, 0]
        x = []
        y = []
        zero = np.zeros((g.nx - 2, g.nz, n))
        x += [np.linspace(x1 - 0.5, x1 + 0.5, n).transpose(1, 2, 0)]
        y += [y0 + 0.5]
        x += [x0 + 0.5]
        y += [np.linspace(y1 + 0.5, y1 - 0.5, n).transpose(1, 2, 0)]
        x += [np.linspace(x1 + 0.5, x1 - 0.5, n).transpose(1, 2, 0)]
        y += [y0 - 0.5]
        x += [x0 - 0.5]
        y += [np.linspace(y1 - 0.5, y1 + 0.5, n).transpose(1, 2, 0)]
        x = np.concatenate([k + zero for k in x], axis=-1)
        y = np.concatenate([k + zero for k in y], axis=-1)
        RZ = np.array(g.getCoordinate(x, y))

        dy = RZ[1, ..., 1:] - RZ[1, ..., :-1]
        xx = (RZ[0, ..., :-1] + RZ[0, ..., 1:]) / 2

        A[1:-1, gi] = -np.sum(xx * dy, axis=-1)
        Ar[1:-1, gi] = -np.sum(0.5 * xx * xx * dy, axis=-1)

        RZs += [RZ]
    RZs = np.array(RZs)
    # 3.160 s:  (4, 2, 6, 32, 200)
    RZs = RZs.transpose(1, 2, 0, 3, 4)

    volume = Ar
    # f =
    #    V = [f[k] for k in ("dx", "dy", "dz", "J")]

    RZ = np.array(
        [
            g.getCoordinate(
                *np.meshgrid(np.arange(g.nx), np.arange(g.nz), indexing="ij")
            )
            for g in pols
        ]
    ).transpose(1, 2, 0, 3)

    RZ = np.array([(g.R, g.Z) for g in pols]).transpose(1, 2, 0, 3)

    # volume = dx * dy * dz * J
    # volume = V[0] * V[1] * V[2] * V[3]
    # A.shape, Ar.shape, RZ.shape

    # AreaXplus
    # Z
    # Λ
    # |   a        b          c
    # |
    # |       ------------
    # |       |          X
    # |       |          X
    # |   h   |    o     X    d
    # |       |          X
    # |       |          X
    # |       ------------
    # |
    # |   g        f          e
    # |
    # |------------------------> x
    #
    # We are calculating the area of the surface denoted by X
    # Note that we need to split it in two parts.

    log("calculating pos ...")
    # pos = RZs[..., :n]
    tmp = list(
        np.meshgrid(
            np.arange(g.nx - 1) + 0.5,
            np.arange(g.nz) - 0.5,
            np.linspace(0, 1, n),
            indexing="ij",
        )
    )
    tmp[1] += tmp[2]

    pos = np.array([g.getCoordinate(*tmp[:2]) for g in pols]).transpose(1, 2, 0, 3, 4)
    log("done")
    # direction of edge, length ~ 1/nx
    dR = pos[..., :-1] - pos[..., 1:]
    dX = np.sqrt(np.sum(dR**2, axis=0))
    area = dX * (pos[0, ..., 1:] + pos[0, ..., :-1]) * 0.5
    assert area.shape[0] > 2
    # Vector in normal direction
    assert dR.shape[0] == 2
    dRr = np.array((-dR[1], dR[0]))
    dRr /= np.sqrt(np.sum(dRr**2, axis=0))
    dRr *= area
    dRr = np.sum(dRr, axis=-1)

    # vector in derivative direction
    dxR = RZ[:, 1:] - RZ[:, :-1]
    dzR = np.roll(RZ, -1, axis=-1) - np.roll(RZ, 1, axis=-1)
    dzR = 0.5 * (dzR[:, 1:] + dzR[:, :-1])

    # dxR /= (np.sum(dxR**2, axis=0))
    # dzR /= (np.sum(dzR**2, axis=0))
    log("starting solve")
    dxzR = np.array((dxR, dzR)).transpose(2, 3, 4, 1, 0)
    coefsX = np.linalg.solve(dxzR, dRr.transpose(1, 2, 3, 0)[..., None])[..., 0]
    log("done")

    # AreaZplus
    # Z
    # Λ
    # |   a        b          c
    # |
    # |       -XXXXXXXXXX-
    # |       |          |
    # |       |          |
    # |   h   |    o     |    d
    # |       |          |
    # |       |          |
    # |       ------------
    # |
    # |   g        f          e
    # |
    # |------------------------> x

    log("concatenate")
    tmp = list(
        np.meshgrid(
            np.arange(g.nx - 2) + 0.5,
            np.arange(g.nz) + 0.5,
            np.linspace(0, 1, n),
            indexing="ij",
        )
    )
    tmp[0] += tmp[2]
    pos = np.array([g.getCoordinate(*tmp[:2]) for g in pols]).transpose(1, 2, 0, 3, 4)

    dR = pos[..., :-1] - pos[..., 1:]
    dX = np.sqrt(np.sum(dR**2, axis=0))
    area = dX * (pos[0, ..., 1:] + pos[0, ..., :-1]) * 0.5
    log("get normal vector")
    # Vector in normal direction
    dRr = np.array((dR[1], -dR[0]))
    dRr /= np.sqrt(np.sum(dRr**2, axis=0))
    dRr *= area
    dRr = np.sum(dRr, axis=-1)
    # vector in derivative direction
    dxR = RZ[:, 2:] - RZ[:, :-2]
    dxR = 0.5 * (np.roll(dxR, -1, axis=-1) + dxR)
    dzR = (np.roll(RZ, -1, axis=-1) - RZ)[:, 1:-1]
    dxzR = np.array((dxR, dzR)).transpose(2, 3, 4, 1, 0)
    log("solving again")
    coefsZ = np.linalg.solve(dxzR, dRr.transpose(1, 2, 3, 0)[..., None])[..., 0]
    log("done")

    test(RZ, volume, coefsX, coefsZ, plot=plot)

    return write(RZ, volume, coefsX, coefsZ)


def test(RZ, volume, coefsX, coefsZ, plot=False):
    inp = np.sin(RZ[1])
    ana = -np.sin(RZ[1])
    if 0:
        inp = RZ[1] ** 2
        ana = RZ[1] ** 0
    if 1:
        inp = np.sin(RZ[0])
        ana = -np.sin(RZ[0]) + np.cos(RZ[0]) / RZ[0]
    if 0:
        inp = RZ[0] ** 3
        ana = 6 * np.sin(RZ[0])

    def xp(ijk):
        return (ijk[0] + 1, *ijk[1:])

    def xm(ijk):
        return (ijk[0] - 1, *ijk[1:])

    def zp(ijk):
        return _zp(*ijk)

    def _zp(i, j, k):
        return (i, j, (k + 1) % zmax)

    def zm(i, j, k):
        return (i, j, (k - 1) % zmax)

    log("testing")
    result = np.zeros(inp.shape)
    results = [np.zeros(inp.shape) for _ in range(4)]
    # dx =
    dz2 = np.roll(inp, -1, axis=-1) - np.roll(inp, 1, axis=-1)
    i, j, k = inp.shape
    zmax = k
    dx = inp[1:] - inp[:-1]
    dz = 0.5 * (dz2[1:] + dz2[:-1])
    this = -(coefsX[..., 0] * dx + coefsX[..., 1] * dz)
    result[:-1] -= this
    result[1:] += this
    for r, t in zip(results, (-coefsX[..., 0] * dx, -coefsX[..., 1] * dz)):
        r[:-1] -= t
        r[1:] += t
    print("expect 0:", np.max(np.abs((result - results[0] - results[1]))))
    if 1:
        dx2 = inp[2:] - inp[:-2]
        dx2 = 0.5 * (np.roll(dx2, -1, axis=-1) + dx2)
        dz2 = (np.roll(inp, -1, axis=-1) - inp)[1:-1]
        t1 = coefsZ[..., 0] * dx2
        t2 = coefsZ[..., 1] * dz2
        this = -(t1 + t2)
        result[1:-1] -= this
        result[1:-1] += np.roll(this, 1, -1)
        for r, t in zip(results[2:], (-t1, -t2)):
            r[1:-1] -= t
            # np.roll(r, -1, -1)[1:-1] += t
            r[1:-1] += np.roll(t, 1, -1)

    result[0] = 0
    result[-1] = 0
    for r in results:
        r[0] = 0
        r[-1] = 0

    if plot:
        import matplotlib.pyplot as plt

        f, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        res = result / volume
        for d, ax, t in zip(
            (res, ana, res - ana),
            axs,
            ("computed", "analytical", "error"),
        ):
            slc = slice(1, -1), 1
            per = np.percentile(d[slc], [0, 2, 98, 100])
            print("Percentiles", per)
            p = ax.pcolormesh(*[k[slc] for k in RZ], d[slc], vmin=per[1], vmax=per[2])
            ax.set_title(t)
            plt.colorbar(p, ax=ax)
        plt.show()
    print("error:", np.mean(l2(result[1:-1] / volume[1:-1] - ana[1:-1])))


def fixup(RZ, d):
    c1 = np.zeros(RZ[0].shape)
    if c1.shape[0] == d.shape[0] + 1:
        c1[:-1] = d
    else:
        c1[1:-1] = d
    # c1 = np.roll(c1, -1, -1)
    return c1


def fixup2(d, val):
    if val.shape[0] == d.shape[0] + 1:
        d[:] = val[:-1]
    else:
        d[:] = val[1:-1]


def write(RZ, volume, coefsX, coefsZ):
    ret = {}
    log("writing")
    for d, x1 in zip((coefsX, coefsZ), "XZ"):
        for i, x2 in enumerate("XZ"):
            key = f"dagp_fv_{x1}{x2}"
            ret[key] = fixup(RZ, d[..., i])
        key = "dagp_fv_volume"
        ret[key] = volume
    log("done")
    print(ret.keys())
    return ret


if __name__ == "__main__":
    for flag, step in (("-v", +1), ("-q", -1)):
        while flag in sys.argv:
            verbose += step
            sys.argv.remove(flag)
    plot = "-p" in sys.argv
    if plot:
        sys.argv.remove("-p")

    for fn in sys.argv[1:]:
        print(fn)
        test(*load(fn), plot=plot)
