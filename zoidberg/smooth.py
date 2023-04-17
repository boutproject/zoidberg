import numpy as np
from scipy import linalg


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


def gen_newr(r, z, splitedlist, bounds):
    newr = r.copy()
    newz = z.copy()
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
        A = np.array(
            [
                [1, 0.5, 1 / 300, 0, 0, 0],
                [0, 0, 0, 1, 0.5, 1 / 300],
                [diff_minus_z, 0, 0, -diff_minus_r, 0, 0],
                [
                    diff_plus_z,
                    diff_plus_z,
                    diff_plus_z / 100,
                    -diff_plus_r,
                    -diff_plus_r,
                    -diff_plus_r / 100,
                ],
            ]
        )
        b = np.array(
            [[r[end_index] - r[start_index]], [z[end_index] - z[start_index]], [0], [0]]
        )
        x = linalg.lstsq(A, b)
        S = np.linspace(0, 1, num=len(alist))
        X = (
            r[start_index]
            + x[0][0] * S
            + (x[0][1] * S**2) * 0.5
            + (x[0][2] * S**3) / 300
        )
        Y = (
            z[start_index]
            + x[0][3] * S
            + (x[0][4] * S**2) * 0.5
            + (x[0][5] * S**3) / 300
        )
        if start_index < 0:
            newr = rollup(newr, start_index, len(alist), X)
            newz = rollup(newz, start_index, len(alist), Y)
        else:
            newr[start_index : start_index + len(alist)] = X
            newz[start_index : start_index + len(alist)] = Y
    return newr, newz


def smooth(r, z, cutoff=0.75, bounds=10):
    """
    Smooth an ordered set of r-z coordinates to avoid sharp edges.
    
    For all points, where the cos of the angle between the point, and the points ± `bounds` are below `cutoff`, the points within the range are replaced by a smoothed set of points.
    
    r : 1d array
    z : 1d array
    cutoff : float
        cos of angle
    bounds :  integer
        distance of which to smooth
        
    Return:
    tuple of 1d-arrays
    The smoothed values, if there where outliers, or the original data otherwise.
    """
    outliers = find_outliers(r, z, cutoff, bounds)
    splitedlist = generate_points_tosmooth(bounds, outliers, len(r))
    return gen_newr(r, z, splitedlist, bounds)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    import zoidberg as zb

    fn = sys.argv[1]
    dat = np.loadtxt(fn)
    dat.shape = (-1, 2, dat.shape[1])
    line = zb.rzline.RZline(*dat[1])
    r, z = dat[1]
    r = []
    z = []

    newr, newz = smooth(r, z, cutoff=0.75, bounds=10)
    plt.plot(newr, newz, "x-")
    plt.plot(r, z)
    plt.show()
