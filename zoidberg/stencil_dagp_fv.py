#!/usr/bin/env python
# coding: utf-8

import time

time_start = time.time()

from boututils.datafile import DataFile as DF
import numpy as np
import sys


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


def doit(fn):
    with DF(fn) as f:
        RZ = [f[k] for k in "RZ"]
    log("opened")
    RZ = np.array(RZ)
    log("read")

    A = None
    nx = 2
    spc = np.linspace(0, 0.5, nx, endpoint=True)
    spc2 = np.linspace(0.5, 1, nx, endpoint=True)

    ### Calculate Volume of the cell
    #
    ### Go in a line around the cell
    # xf is the normalised coordinates in x
    # yf is the normalised coordinates in z
    # i and j are the offset in x and z
    # i,j = 0 -> in x and z in lower direction
    for xf, yf, i, j in (
        (spc2, 0.5, 1, 1),
        (0.5, spc2[::-1], 1, 1),
        (0.5, spc[::-1], 1, 0),
        (spc2[::-1], 0.5, 1, 0),
        (spc[::-1], 0.5, 0, 0),
        (0.5, spc, 0, 0),
        (spc, 0.5, 0, 1),
        (0.5, spc2, 0, 1),
    ):
        log(f"doing offset ({i}, {j})")
        dx = np.arange(2) + i
        dy = np.arange(2) + j

        def slc(i):
            i1 = -2 + i
            return slice(i, i1 or None)

        xii = [
            [np.roll(RZ[:, slc(i)], -j + 1, axis=-1)[..., None] for i in dy] for j in dx
        ]
        xa, xb = xii
        x00, x01 = xa
        x10, x11 = xb
        x = yf * (xf * x00 + (1 - xf) * x10) + (1 - yf) * (xf * x01 + (1 - xf) * x11)
        y = x[1]
        dy = (y[..., 1] - y[..., 0])[..., None]
        xx = (x[0, ..., :-1] + x[0, ..., 1:]) / 2
        if A is None:
            A = np.zeros(xx[..., 0].shape)
            Ar = np.zeros(xx[..., 0].shape)
        A -= np.sum(xx * dy, axis=-1)
        Ar -= np.sum(0.5 * xx * xx * dy, axis=-1)
        # plt.plot(x[0], x[1], "gx")
    volume = Ar
    A.shape, Ar.shape, RZ.shape

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
    # Note that we need to split it in two parts. Maybe?
    nx = 2
    spc3 = np.linspace(0, 1, nx)
    # pos goes in +z direction
    # First segments starts at the centre of b,c,d,o
    # and goes to centre of o,d
    startpoint = toCent(RZ)
    midpoint = (RZ[:, :-1] + RZ[:, 1:]) / 2
    endpoint = np.roll(startpoint, -1, -1)

    log("calculating pos ...")
    pos = np.concatenate(
        [
            (1 - spc3[:-1]) * startpoint[..., None] + spc3[:-1] * midpoint[..., None],
            (1 - spc3) * midpoint[..., None] + spc3 * endpoint[..., None],
        ],
        axis=-1,
    )
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
    print(dRr.shape, area.shape)
    dRr *= area
    dRr = np.sum(dRr, axis=-1)
    print(np.nanmean(dRr))

    # vector in derivative direction
    dxR = RZ[:, 1:] - RZ[:, :-1]
    print("Maximum radial grid distance:", np.max(l2(dxR)))
    dzR = np.roll(RZ, -1, axis=-1) - np.roll(RZ, 1, axis=-1)
    dzR = 0.5 * (dzR[:, 1:] + dzR[:, :-1])

    # dxR /= (np.sum(dxR**2, axis=0))
    # dzR /= (np.sum(dzR**2, axis=0))
    log("starting solve")
    dxzR = np.array((dxR, dzR)).transpose(2, 3, 4, 1, 0)
    coefs = np.linalg.solve(dxzR, dRr.transpose(1, 2, 3, 0))
    log("done")

    areaX = area
    coefsX = coefs

    # In[154]:

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
    nx = 3
    spc3 = np.linspace(0, 1, nx)
    cent = np.roll(toCent(RZ), -1, -1)
    startpoint = cent[:, :-1]  # a
    midpoint = (RZ[:, 1:-1] + np.roll(RZ[:, 1:-1], -1, -1)) / 2  # b
    endpoint = cent[:, 1:]  # c
    log("concatenate")
    pos = np.concatenate(
        [
            (1 - spc3[:-1]) * startpoint[..., None] + spc3[:-1] * midpoint[..., None],
            (1 - spc3) * midpoint[..., None] + spc3 * endpoint[..., None],
        ],
        axis=-1,
    )
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
    coefs = np.linalg.solve(dxzR, dRr.transpose(1, 2, 3, 0))
    log("done")
    coefs.shape

    areaZ = area
    coefsZ = coefs

    # In[155]:

    inp = np.sin(RZ[1])
    ana = -np.sin(RZ[1])  # + np.cos(RZ[0])/RZ[0]
    if 0:
        inp = RZ[1] ** 2
        ana = RZ[1] ** 0
    if 1:
        inp = np.sin(RZ[0])
        ana = -np.sin(RZ[0]) + np.cos(RZ[0]) / RZ[0]
    if 0:
        inp = RZ[0] ** 3
        ana = 6 * np.sin(RZ[0])

    # In[156]:

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
    for r, t in zip(results, (coefsX[..., 0] * dx, coefsX[..., 1] * dz)):
        r[:-1] -= t
        r[1:] += t
    results[0]

    if 1:
        dx2 = inp[2:] - inp[:-2]
        dx2 = 0.5 * (np.roll(dx2, -1, axis=-1) + dx2)
        dz2 = (np.roll(inp, -1, axis=-1) - inp)[1:-1]
        t1 = coefsZ[..., 0] * dx2
        t2 = coefsZ[..., 1] * dz2
        this = -(t1 + t2)
        result[1:-1] -= this
        result[1:-1] += np.roll(this, 1, -1)
        for r, t in zip(results[2:], (t1, t2)):
            r[1:-1] -= t
            np.roll(r, -1, -1)[1:-1] += t

    result[0] = 0
    for r in results:
        r[0] = 0
        r[-1] = 0

    print(fn, "error:", np.mean(l2(result[1:-1] / volume - ana[1:-1])))

    def fixup(d):
        c1 = np.zeros(RZ[0].shape)
        if c1.shape[0] == d.shape[0] + 1:
            c1[:-1] = d
        else:
            c1[1:-1] = d
        # c1 = np.roll(c1, -1, -1)
        return c1

    log("writing")
    with DF(fn, write=True) as f:
        for d, x1 in zip((coefsX, coefsZ), "XZ"):
            for i, x2 in enumerate("XZ"):
                key = f"dagp_fv_{x1}{x2}"
                f.write(key, fixup(d[..., i]))
        key = "dagp_fv_volume"
        f.write(key, fixup(volume))
    log("done")


if __name__ == "__main__":
    for flag, step in (("-v", +1), ("-q", -1)):
        while flag in sys.argv:
            verbose += step
            sys.argv.remove(flag)

    for fn in sys.argv[1:]:
        print(fn)
        doit(fn)
