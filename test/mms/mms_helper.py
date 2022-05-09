import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import boutcore as bc

lst = [4, 8, 16, 32, 64]


def extend(p):
    # print(p.shape, p.dims)
    out = np.concatenate([p[:, -1:, :], p, p[:, :1, :]], axis=1)
    # print(out.shape)
    return out


def diff(p, axis):
    if axis == 0:
        out = np.zeros_like(p)
        out[:-1] = p[1:] - p[:-1]
        out[1:] += p[1:] - p[:-1]
        out[1:-1] /= 2
        return out
    return (np.roll(p, -1, axis=axis) - np.roll(p, 1, axis=axis)) / 2


def diff2(p, axis):
    assert axis != 0
    return (2 * p - np.roll(p, 1, axis=axis) - np.roll(p, -1, axis=axis)) / 2


def clean(p):
    return p[:, 1:-1, :]


def test(fn, inpf, anaf, testfunc):
    grid = xr.open_dataset(fn)
    bc.Options().set("mesh:file", fn, force=True)
    mesh = bc.Mesh(section="")

    f = bc.create3D("0", mesh)

    inp = inpf(grid)
    ana = anaf(grid)

    # Set input field
    f[:, :, :] = inp
    mesh.communicate(f)
    calc = testfunc(f).get()
    l2 = np.sqrt(np.mean(clean(ana - calc)[1:-1] ** 2))

    if 0:
        fig, axs = plt.subplots(1, 3)
        calcd = clean(calc)[:, 1]
        rz = [clean(extend(grid[d]))[:, 1] for d in "RZ"]
        anad = ana[:, 1]
        for dat, label, ax in zip(
            [anad, calcd, (calcd - anad)], ["ana", "calc", "err"], axs
        ):
            plot = ax.pcolormesh(rz[0][1:-1, 1:-1], rz[1][1:-1, 1:-1], dat[1:-1, 1:-1])
            ax.set_title(fn[-20:] + " " + label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plot, cax=cax, orientation="vertical")
    return l2


def do_tests(grids, inpf, anaf, testf, expect=2):
    fail = False
    for mode, todo in grids.items():
        l2 = [test(x[0], inpf, anaf, testf) for x in todo]
        lst = [x[-1] for x in todo]

        errc = np.log(l2[-2] / l2[-1])
        difc = np.log(lst[-1] / lst[-2])
        conv = errc / difc
        print(mode, conv, l2)
        if not np.isclose(conv, l2, atol=0.1):
            fail = True
        plt.show()
    assert not fail
