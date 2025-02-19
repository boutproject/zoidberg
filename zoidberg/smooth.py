import numpy as np
from scipy import linalg
from scipy.optimize import least_squares


def find_outliers(r, z, cutoff=0.75, bounds=10):
    ### find points in distance of 10 from each other, which create a too sharp pseudoangle
    outliers = []
    ### go through points with 10 distance
    for item in range(len(z)):
        point1 = np.array([z[item], r[item]])
        point2 = np.array([z[item - bounds], r[item - bounds]])
        point3 = np.array([z[item - bounds * 2], r[item - bounds * 2]])
        ### calculate pseudoangle without artan
        a = point1 - point2
        b = point1 - point3
        product = np.dot(a, b)
        magn_a = np.sqrt(a.dot(a))
        magn_b = np.sqrt(b.dot(b))
        pseudoangle = product / (magn_a * magn_b)
        if pseudoangle < cutoff:
            outliers.append(item - bounds)
    return outliers


def generate_points_tosmooth(bounds, outliers, length):
    ### for every position to be smoothed, get all points within a range surrouding that point
    # assert len(outliers) != 0
    if not outliers:
        return []
    todel = set()
    for dup in outliers:
        todel.update([x for x in range(dup - bounds, dup + bounds + 1) if x < length])
    ###
    todel = sorted(list(todel))
    splitedlist = []
    first = 0
    last = todel[0]
    for index, value in enumerate(todel[1:]):
        if last + 1 != value:
            splitedlist.append(todel[first : index + 1])
            first = index + 1
        last = value
    splitedlist.append(todel[first:])
    return splitedlist


def rollup(original_array, startindex, length, replacement):
    original_array = np.roll(original_array, -startindex)
    original_array[0:length] = replacement
    original_array = np.roll(original_array, startindex)
    return original_array


def gen_newr(r, z, splitedlist, bounds, plot):
    newr = r.copy()
    newz = z.copy()
    have_failed = False
    deb = []
    for alist in splitedlist:
        assert len(alist) >= bounds
        if len(alist) >= len(r) / 2:
            raise ValueError("Too many points to smooth")
        start_index = alist[0]
        end_index = alist[-1]
        diff_minus_r = r[start_index] - r[start_index - 1]
        diff_minus_z = z[start_index] - z[start_index - 1]
        if end_index < len(r) - 1:
            seconds = end_index + 1
        else:
            seconds = 0
        diff_plus_r = r[end_index] - r[seconds]
        diff_plus_z = z[end_index] - z[seconds]
        fac = 1
        if 0:
            A = np.array(
                [
                    [1, 0.5, 1 / 3 / fac, 1 / 4, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0.5, 1 / 3 / fac, 1 / 4],
                    [diff_minus_z, 0, 0, 0, -diff_minus_r, 0, 0, 0],
                    [
                        diff_plus_z,
                        diff_plus_z,
                        diff_plus_z / fac,
                        diff_plus_z,
                        -diff_plus_r,
                        -diff_plus_r,
                        -diff_plus_r / fac,
                        -diff_plus_r,
                    ],
                ]
            )
        else:
            A = np.array(
                [
                    [1, 0.5, 1 / 3 / fac, 0, 0, 0],
                    [0, 0, 0, 1, 0.5, 1 / 3 / fac],
                    [diff_minus_z, 0, 0, -diff_minus_r, 0, 0],
                    [
                        diff_plus_z,
                        diff_plus_z,
                        diff_plus_z / fac,
                        -diff_plus_r,
                        -diff_plus_r,
                        -diff_plus_r / fac,
                    ],
                ]
            )

        b = np.array(
            [r[end_index] - r[start_index], z[end_index] - z[start_index], 0, 0]
        )

        def lenofs(x):
            if len(x) == 6:
                x1r, x2r, x3r = x[:3]
                x1z, x2z, x3z = x[3:]
                xr = x[:3]
                xz = x[3:]
                xs = xr * xr + xz * xz
                x1s, x2s, x3s = xs
                x12 = x1r * x2r + x1z * x2z
                x13 = x1r * x3r + x1z * x3z
                x23 = x2r * x3r + x2z * x3z
                num = 100
                s = np.linspace(0, 1, num)
                tmp = np.sqrt(
                    x1s
                    + x2s * s**2
                    + x3s * s**4
                    + 2 * (x12 * s + x13 * s**2 + x23 * s**3)
                )
                return np.sum(tmp) / num
            x1r, x2r, x3r, x4r = x[:4]
            x1z, x2z, x3z, x4z = x[4:]
            xr = x[:4]
            xz = x[4:]
            xs = xr * xr + xz * xz
            x1s, x2s, x3s, x4s = xs
            x12 = x1r * x2r + x1z * x2z
            x13 = x1r * x3r + x1z * x3z
            x14 = x1r * x4r + x1z * x4z
            x23 = x2r * x3r + x2z * x3z
            x24 = x2r * x4r + x2z * x4z
            x34 = x3r * x4r + x3z * x4z
            # return x1s + x12 + 2 / 3 * x13 + 1/2*x14 +1/3 * x23 + 1 / 4 * x2s + 1 / 5 * x3s +1/7*x4s
            num = 100
            s = np.linspace(0, 1, num)
            tmp = np.sqrt(
                x1s
                + x2s * s**2
                + x3s * s**4
                + x4s * s**6
                + 2
                * (
                    x12 * s
                    + x13 * s**2
                    + x14 * s**3
                    + x23 * s**3
                    + x24 * s**4
                    + x34 * s**5
                )
            )
            return np.sum(tmp) / num

        def mysin(a, b):
            aa, ab = a
            ba, bb = b
            return (aa * bb - ab * ba) / np.sqrt(
                (aa * aa + ab * ab + 1e-8) * (ba * ba + bb * bb + 1e-8)
            )

        def mycos(a, b):
            aa, ab = a
            ba, bb = b
            return (aa * ba + ab * bb) / np.sqrt(
                (aa * aa + ab * ab + 1e-8) * (ba * ba + bb * bb + 1e-8)
            )

        def curv(x, f):
            k = len(x) // 2
            a = np.sum(x[:k] * np.arange(k))
            b = x[1]
            c = np.sum(x[k:] * np.arange(k))
            d = x[k + 1]
            return (a * a + c * c) * f, (b * b + d * d) * f

        def fun(x, A, b, signs):
            ret = A @ x - b
            # k = len(x) // 2
            # newsigns = (x[0], x[1], np.sum(x[:k]), np.sum(x[k:]))
            # mycoss = mycos(signs[:2], newsigns[:2]), mycos(signs[2:], newsigns[2:])
            return (
                *ret,
                lenofs(x) / 1e5,
                # (1 - mycoss[0]) / 1e2,
                # (mycoss[1] - 1) / 1e2,
                *(curv(x, 1 / 1e3)),
            )

        if 0:
            x = linalg.lstsq(A, b)
            x0 = x[0]
        else:
            signs = (diff_minus_r, diff_minus_z, -diff_plus_r, -diff_plus_z)
            x = least_squares(
                fun,
                np.zeros(A.shape[1]),
                args=(A, b, signs),
                gtol=None,
                xtol=1e-14,
                ftol=None,
            )
            # if np.max(np.abs(x.fun)) > 1e-5:
            if x.optimality > 1e-8:
                have_failed = True
                deb += [[x.cost, x.optimality, np.max(np.abs(x.fun))]]
                continue
            x0 = x.x
            newsigns = (x0[0], x0[1], np.sum(x0[:3]), np.sum(x0[3:]))

        S = np.linspace(0, 1, num=len(alist))

        if len(x0) == 6:
            X = (
                r[start_index]
                + x0[0] * S
                + (x0[1] * S**2) * 0.5
                + (x0[2] * S**3) / 3 / fac
            )
            Y = (
                z[start_index]
                + x0[3] * S
                + (x0[4] * S**2) * 0.5
                + (x0[5] * S**3) / 3 / fac
            )
        else:
            X = (
                r[start_index]
                + x0[0] * S
                + (x0[1] * S**2) * 0.5
                + (x0[2] * S**3) / 3 / fac
                + x0[3] * S**4 / 4
            )
            Y = (
                z[start_index]
                + x0[4] * S
                + (x0[5] * S**2) * 0.5
                + (x0[6] * S**3) / 3 / fac
                + x0[7] * S**4 / 4
            )

        # Check spacing at beginning and end
        dstart = (X[0] - newr[start_index - 1]) ** 2 + (
            Y[0] - newz[start_index - 1]
        ) ** 2
        dend = (X[-1] - newr[start_index + len(alist)]) ** 2 + (
            Y[-1] - newz[start_index + len(alist)]
        ) ** 2
        maxfac = 3
        if not (maxfac**-2 < dstart / dend < maxfac**2):
            print(f"dstart / dend is {dstart/dend} - check has failed")
            deb += [[dstart, dend, 3]]
            have_failed = True

        if start_index < 0:
            newr = rollup(newr, start_index, len(alist), X)
            newz = rollup(newz, start_index, len(alist), Y)
        else:
            newr[start_index : start_index + len(alist)] = X
            newz[start_index : start_index + len(alist)] = Y
        if plot:
            from matplotlib.collections import LineCollection

            S = np.linspace(0, 1, 1000)

            X = (
                r[start_index]
                + x0[0] * S
                + (x0[1] * S**2) * 0.5
                + (x0[2] * S**3) / 3 / fac
            )
            Y = (
                z[start_index]
                + x0[3] * S
                + (x0[4] * S**2) * 0.5
                + (x0[5] * S**3) / 3 / fac
            )
            points = np.array([X, Y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(S.min(), S.max())
            lc = LineCollection(segments, cmap="viridis", norm=norm)
            lc.set_array(S)
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)
            # plt.plot(X, Y, "k-")

    return newr, newz, have_failed, deb


alldeb = []


def smooth(r, z, cutoff=0.75, bounds=10, bounds_find=None, plot=False):
    """
    Smooth an ordered set of r-z coordinates to avoid sharp edges.

    For all points, where the cos of the angle between the point, and the points ± `bounds` are below `cutoff`, the points within the range are replaced by a smoothed set of points.

    r : 1d array
    z : 1d array
    cutoff : float
        cos of angle
    bounds :  integer
        distance of which to smooth
    bounds_find : integer (optional)
        if given, distance in which to check for edges, bounds otherwise

    Return:
    tuple of 1d-arrays
    The smoothed values, if there where outliers, or the original data otherwise.
    """
    if plot:
        import matplotlib.pyplot as plt

        # print(r, z)
        plt.plot(r, z, "o-", label="input")

    bounds_find = bounds_find or bounds
    bounds0 = bounds
    debs = []
    while True:
        outliers = []
        for b in range(1, bounds_find + 1):
            # print(find_outliers(r, z, cutoff, b))
            outliers += find_outliers(r, z, cutoff, b)
        if plot:
            # print(np.array([(r[i], z[i]) for i in outliers]).T)
            if outliers:
                plt.plot(
                    *np.array([(r[i], z[i]) for i in outliers]).T, "x", label="outlier"
                )
        splitedlist = generate_points_tosmooth(bounds, outliers, len(r))
        if plot:
            # plt.plot(
            # print(splitedlist)
            pass
        a, b, fail, deb = gen_newr(r, z, splitedlist, bounds, plot)
        if not fail:
            ret = a, b
            break
        r, z = a, b
        bounds += 1
        debs += [deb[-1]]
        if bounds > bounds0 * 5:
            plt.legend()
            plt.gca().set_aspect("equal")
            plt.title("failed")
            alldeb.append(debs)
            return

    alldeb.append(debs)
    if plot:
        plt.plot(*ret, "o-", label="new")
        plt.legend()
        plt.gca().set_aspect("equal")
        # plt.show()
    return ret


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    fn = sys.argv[1]
    dat = np.loadtxt(fn)
    dat.shape = (-1, 2, dat.shape[1])
    j = 0
    for i in range(len(dat)):
        r, z = dat[i]
        if find_outliers(r, z, 0.75, 10):
            plt.figure()
            smooth(r, z, cutoff=0.75, bounds=10, plot=True)
            j += 1
            print(f"figure {j}")
            # if j == 2:
            #    break

    if any([len(x) for x in alldeb]):
        f, axs = plt.subplots(3, 1)
        for deb in alldeb:
            x = np.array(deb)
            print(deb)
            print(x.shape)
            for a, b in zip(axs, x.T):
                a.plot(b)
    plt.show()
