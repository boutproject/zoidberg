from mms_helper import extend, do_tests, plt, bc, np
from radial_grid import grids


set1 = (
    lambda grid: np.sin(extend(grid.r_minor)),
    lambda grid: np.cos(extend(grid.r_minor)),
)

set2 = (
    lambda grid: extend(grid.r_minor) ** 3,
    lambda grid: 3 * extend(grid.r_minor) ** 2,
)


if __name__ == "__main__":
    bc.init("-d mms -q -q -q")

    do_tests(grids, *set1, bc.DDX)
    do_tests(grids, *set2, bc.DDX)

    plt.show()
