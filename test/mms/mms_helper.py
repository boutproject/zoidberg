import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import boutpp as bc
import boutconfig

assert boutconfig.isMetric3D(), f"""Require 3D metric - but boutpp seems to be compiled with 2D Metrics
NB: boutpp     is {bc.__path__}
    boutconfig is {boutconfig.__path__}"""

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


def test(fn, inpf, anaf, testfunc, iname):
    grid = xr.open_dataset(fn)
    bc.Options.root().set("mesh:file", fn, force=True)
    mesh = bc.Mesh(section="")

    f = bc.create3D("0", mesh)

    inp = inpf(grid)
    ana = anaf(grid)

    # Set input field
    f[:, :, :] = inp
    mesh.communicate(f)
    calc = testfunc(f).get()
    l2 = np.sqrt(np.mean(clean(ana - calc)[1:-1] ** 2))

    if 1:
        fig, axs = plt.subplots(1, 4)
        calcd = clean(calc)[:, 1]
        rz = [clean(extend(grid[d]))[:, 1] for d in "RZ"]
        anad = ana[:, 1]
        for dat, label, ax in zip(
            [clean(inp)[:, 1], anad, calcd, (calcd - anad)],
            ["inp", "ana", "calc", "err"],
            axs,
        ):
            plot = ax.pcolormesh(rz[0][1:-1, 1:-1], rz[1][1:-1, 1:-1], dat[1:-1, 1:-1])
            ax.set_title(fn[-20:] + " " + label + " " + iname)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plot, cax=cax, orientation="vertical")
    return l2


def do_tests(grids, inpf, anaf, iname, testf, expect=2):
    fail = False
    for mode, todo in grids.items():
        l2 = [test(x[0], inpf, anaf, testf, iname) for x in todo]
        lst = [x[-1] for x in todo]

        errc = np.log(l2[-2] / l2[-1])
        difc = np.log(lst[-1] / lst[-2])
        conv = errc / difc
        info = todo[0][0]
        info = "_".join(info.split("_")[:2])
        print(info, mode, todo, conv, l2)
        with open(f"result_{info}_{iname}.txt", "w") as f:
            f.write(info)
            f.write("\n")
            f.write(iname)
            f.write("\n")
            f.write(" ".join([str(x) for x in lst]))
            f.write("\n")
            f.write(" ".join([str(x) for x in l2]))
            f.write("\n")
        passes = np.isclose(conv, expect, atol=0.2)
        if not passes:
            fail = True
        print(
            f"{conv} {'=' if passes else '!'}= {expect} - {'passing :-)' if passes else 'failing!'}"
        )
        if passes:
            plt.close("all")
        plt.show()
    assert not fail
