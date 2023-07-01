from mms_helper import extend, do_tests, plt, bc, np
from laplace_grid import grids


sets = [
    # (
    #     lambda grid: np.sin(extend(grid["one"] * grid.phi) * 5),
    #     lambda grid: -25 * np.sin(extend(grid["one"] * grid.phi) * 5),
    # ),
    (
        lambda grid: extend(grid.R),
        lambda grid: 1 / extend(grid.R),
        "R",
    ),
    (
        lambda grid: extend(grid.R) ** 2,
        lambda grid: 4 + 0 * extend(grid.R),
        "R²",
    ),
    (
        lambda grid: np.sin(extend(grid.Z)),
        lambda grid: -np.sin(extend(grid.Z)),
        "sin(Z)",
    ),
    (
        lambda grid: np.sin(extend(grid.R)),
        lambda grid: -np.sin(extend(grid.R)) + np.cos(extend(grid.R)) / extend(grid.R),
        "sin(R)",
    ),
    (
        lambda grid: extend(grid.R) ** 3,
        lambda grid: 9 * extend(grid.R),
        "R³",
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
