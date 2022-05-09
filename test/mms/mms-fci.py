import boutcore as bc
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

lst = [4, 8, 16, 32, 64][:-1]


def extend(p):
    return np.concatenate([p[:, -1:, :], p, p[:, :1, :]], axis=1)


def diff(p, axis):
    if axis == 0:
        out = np.zeros_like(p)
        out[:-1] = p[1:] - p[:-1]
        out[1:] += p[1:] - p[:-1]
        out[1:-1] /= 2
        return out
    return np.roll(p, 1, axis=axis) - p


def diff2(p, axis):
    assert axis != 0
    return (2 * p - np.roll(p, 1, axis=axis) - np.roll(p, -1, axis=axis)) / 2


def clean(p):
    return p[:, 1:-1, :]


def fixup(fn):
    grid = xr.open_dataset(fn)
    R = grid.R
    Z = grid.Z
    dims = Z.dims
    one = np.ones_like(Z)

    # grid["dx"] = dims, np.sqrt((diff(R, 0)) ** 2 + (diff(Z, 0)) ** 2)
    # grid.dx[:, 1, :].plot()
    # plt.figure()
    dxn = np.sqrt((diff(R, 0)) ** 2 + (diff(Z, 0)) ** 2)
    dxn /= np.mean(dxn) * R.shape[0]

    # plt.imshow(dxn[:, 1])
    # plt.colorbar()
    # plt.show()

    dzn = np.sqrt((diff(R, 2)) ** 2 + (diff(Z, 2)) ** 2)
    dzn /= np.mean(dzn) * R.shape[2] / 2 / np.pi
    # grid["dz"] = np.sqrt((diff(R, 2)) ** 2 + (diff(Z, 2)) ** 2)
    ny = R.shape[1]
    dy = 2 * np.pi / 5 / ny
    phi = np.linspace(0, np.pi / 5, ny * 5, False)[None, :ny, None] * one + dy / 2
    # grid["dy"] = dims, dy * one

    # grid["dx"] = dims, dxn
    # grid["dy"] = dims, dy * one
    # grid["dz"] = dzn
    # grid["phi"] = (dims, phi)

    # J = np.array([[diff(o, d) for d in range(3)] for o in [R, Z, phi]])

    # g = np.array(
    #     [[[J[k, i] * J[k, j] for k in range(3)] for i in range(3)] for j in range(3)]
    # )
    # g = g.sum(axis=2)
    # g = g.transpose((2, 3, 4, 0, 1))
    # J = J.transpose((2, 3, 4, 0, 1))
    # # g
    # assert np.all(J[..., 0, 1] == diff(R, 1))
    # # Jinv = np.linalg.inv(J)
    # bad = np.linalg.det(g) == 0
    # for i, b in enumerate(bad):
    #     if np.any(b):
    #         plt.figure()
    #         # plt.imshow(np.linalg.det(g)[i])
    #         plt.imshow(b)
    #         plt.colorbar()
    #         plt.title(str(i))
    # plt.show()

    # #    raise
    # ginv = np.linalg.inv(g)
    # det = np.linalg.det(J)
    # if np.min(det) * np.max(det) < 0:
    #     plt.imshow(g[:, 2, :, 0, 0])
    #     plt.colorbar()
    #     plt.show()

    # for i in range(3):
    #     for j in range(3):
    #         grid[f"g_{i+1}{j+1}"] = dims, g[..., i, j]
    #         grid[f"g{i+1}{j+1}"] = dims, ginv[..., i, j]
    g = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            k = f"g{i+1}{j+1}"
            if k in grid:
                g[i, j] = grid[k]
            else:
                g[i, j] = 0 * one
    g = np.array([list(y) for y in g]).transpose((2, 3, 4, 0, 1))
    assert np.all(np.linalg.det(g) > 0)
    for _ in range(6):
        d = diff2(g, 2) + diff2(g, 1)
        fac = 10
        if not np.all(np.linalg.det(g - d / fac) > 0):
            fac = fac * 2
        print(fac)
        g -= d / fac
    assert np.all(np.linalg.det(g) > 0)
    ginv = np.linalg.inv(g)
    # print(g.shape)

    # raise
    # for k in f"g_{i+1}{j+1}", :
    #     if k not in grid:
    #         grid[k] = dims, 0 * one

    for i in range(3):
        for j in range(3):
            for k, d in zip([f"g_{i+1}{j+1}", f"g{i+1}{j+1}"], [ginv, g]):
                grid[k] = grid.g11.dims, d[..., i, j]
                # grid[k] -= diff2(grid[k].data, 2) / 10
    # Force recalculation
    del grid["J"]
    # grid["J"] = grid.g11.dims, np.linalg.det(g)
    # grid.J.data[grid.J.data < 0] = 0
    # for i in range(3):
    #     grid.J.data -= diff2(grid.J.data, 2) / 10
    #     # plt.plot(grid[k][2, 2], "r")
    # # plt.show()
    # assert np.all(grid.J > 0)

    grid.to_netcdf(f"{fn[:-3]}.fix.nc")


def test(fn):
    grid = xr.open_dataset(fn)
    bc.Options().set("mesh:file", fn, force=True)
    mesh = bc.Mesh(section="")
    print(bc.__file__)
    print()
    print()
    print()
    print()
    f = bc.create3D("0", mesh)
    R = extend(grid.R)
    Z = extend(grid.Z)
    one = np.ones_like(Z)

    dx = np.sqrt((diff(R, 0)) ** 2 + (diff(Z, 0)) ** 2)
    dz = np.sqrt((diff(R, 2)) ** 2 + (diff(Z, 2)) ** 2)
    ny = R.shape[1] - 2
    dy = 2 * np.pi / 5 / ny
    phi = np.linspace(0, np.pi / 5, ny * 5, False)[None, : ny + 2, None] - dy * 1.5

    coords = mesh.coordinates
    print(dx.shape)
    # coords.dx[:, :, :] = dx
    # coords.dy[:, :, :] = dy * one
    # coords.dz[:, :, :] = dz

    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    inp = np.sin(X)  # * np.sin(Z) + np.cos(Y)
    # ana = -2 * np.sin(X) * np.sin(Z) - np.cos(Y)
    ana = -np.sin(X)

    # Set input field
    f[:, :, :] = inp
    mesh.communicate(f)
    calc = bc.Laplace(f).get()
    l2 = np.sqrt(np.mean(clean(ana - calc)[2:-2, 1:-1] ** 2))
    ## print(fn, l2)
    #
    fig, axs = plt.subplots(1, 3)
    calcd = clean(calc)[:, 1]
    anad = ana[:, 1]
    for dat, label, ax in zip(
        [anad, calcd, (calcd - anad)], ["ana", "calc", "err"], axs
    ):
        plot = ax.imshow(dat[2:-2, 1:-1])
        ax.set_title(fn[-20:] + " " + label)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax, orientation="vertical")
    # plt.figure()
    # plt.imshow((calcd - anad)[1:-1])
    # plt.title(fn + " err")
    # plt.colorbar()
    return l2


if __name__ == "__main__":
    for x in lst[0:]:
        fixup(f"rotating-ellipse.{x}x{x}x{x}.fci.nc")

    bc.init("-d mms -q -q -q")

    l2 = [test(f"rotating-ellipse.{x}x{x}x{x}.fci.fix.nc") for x in lst[-3:]]

    print(l2)

    plt.show()
# for x in lst:
