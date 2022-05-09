import boutcore as bc
import matplotlib.pyplot as plt
import numpy as np

from mms_helper import extend, do_tests
from poloidal_grid import grids


def inp(grid):
    t = extend(grid.theta)
    return np.sin(t)


def ana(grid):
    t = extend(grid.theta)
    return np.cos(t)


if __name__ == "__main__":
    bc.init("-d mms -q -q -q")

    do_tests(grids, inp, ana, bc.DDZ)

    plt.show()
