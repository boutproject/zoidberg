from mms_helper import extend, do_tests, plt, bc, np
from parallel_grid import grids


set1 = (
    lambda grid: np.sin(extend(grid["one"] * grid.phi) * 5),
    lambda grid: 5 * np.cos(extend(grid["one"] * grid.phi) * 5),
    "sin(phi)",
)
set2 = (
    lambda grid: np.sin(extend(grid["one"] * grid.phi + grid.r_minor + grid.theta) * 5)
    * extend(np.cos(grid.theta)),
    lambda grid: 5
    * np.cos(extend(grid["one"] * grid.phi + grid.r_minor + grid.theta) * 5)
    * extend(np.cos(grid.theta)),
    "sin(phi*r*theta)*cos(theta)",
)


if __name__ == "__main__":
    bc.init("-d mms")  # -q -q -q")

    do_tests(grids, *set1, bc.DDY)
    do_tests(grids, *set2, bc.DDY)

    plt.show()
