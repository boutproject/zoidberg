from mms_helper import extend, do_tests, plt, bc, np
from delp_grid import grids


sets = [
    # (
    #     lambda grid: np.sin(extend(grid["one"] * grid.phi) * 5),
    #     lambda grid: -25 * np.sin(extend(grid["one"] * grid.phi) * 5),
    # ),
    (
        lambda grid: extend(grid.r_minor) ** 3,
        lambda grid: 6 * extend(grid.r_minor),
    ),
    (
        lambda grid: np.cos(extend(grid.theta)),
        lambda grid: -np.cos(extend(grid.theta)),
    ),
    (
        lambda grid: np.sin(extend(grid.Z)),
        lambda grid: -np.sin(extend(grid.Z)),
    ),
    (
        lambda grid: np.sin(extend(grid.R)),
        lambda grid: -np.sin(extend(grid.R)),
    ),
]
# set2 = (
#     lambda grid: np.sin(extend(grid["one"] * grid.phi + grid.r_minor + grid.theta) * 5)
#     * extend(np.cos(grid.theta)),
#     lambda grid: 5
#     * np.cos(extend(grid["one"] * grid.phi + grid.r_minor + grid.theta) * 5)
#     * extend(np.cos(grid.theta)),
# )


if __name__ == "__main__":
    bc.init("-d mms -q -q -q")

    for seti in sets:
        do_tests(grids, *seti, bc.Laplace)
    # do_tests(grids, *set2, bc.DDX)

    plt.show()
