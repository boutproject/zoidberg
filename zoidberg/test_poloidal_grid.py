import numpy as np
import pytest
import shapely

from . import poloidal_grid, rzline


def test_out_of_domain():
    # Create inner and outer boundaries
    inner = rzline.circle(R0=0.0, r=1.0, n=20)
    outer = rzline.circle(R0=0.0, r=2.0, n=20)

    grid = poloidal_grid.grid_elliptic(inner, outer, 10, 10)

    # Test a point in domain

    x, y = grid.findIndex(1.0, 1.0)
    r, z = grid.getCoordinate(x, y)
    assert np.allclose(r, 1.0) and np.allclose(z, 1.0)  # Should get original R,Z

    # Test out of domain

    x, y = grid.findIndex(2.5, 0)
    assert np.allclose(x, 10)  # Should mark out of domain at outer x

    x, y = grid.findIndex(0.5, 0.2)
    assert np.allclose(x, -1)  # Should mark out of domain at inner x

    # Test a mix of values inside and outside domain

    x, y = grid.findIndex([2.5, 1.0], [0.0, 0.5])

    assert np.allclose(x[0], 10)  # First one out of domain

    r, z = grid.getCoordinate(x[1], y[1])
    assert np.allclose(r, 1.0) and np.allclose(z, 0.5)  # Second point in domain


def lobed_line(R0=3.0, a=1.2, amp=0.5, m=3, n=400):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = a * (1 + amp * np.cos(m * theta))
    return rzline.RZline(R0 + r * np.cos(theta), r * np.sin(theta))


def cell_areas(R, Z):
    """Signed areas of the cell corners' triangles, smallest per cell.
    Negative values mean that the cell is folded or not convex."""
    R = np.concatenate([R, R[:, :1]], axis=1)
    Z = np.concatenate([Z, Z[:, :1]], axis=1)
    corners = [(R[:-1, :-1], Z[:-1, :-1]), (R[1:, :-1], Z[1:, :-1])]
    corners += [(R[1:, 1:], Z[1:, 1:]), (R[:-1, 1:], Z[:-1, 1:])]
    areas = []
    for i in range(4):
        (r0, z0), (r1, z1), (r2, z2) = (corners[(i + j) % 4] for j in (-1, 0, 1))
        areas.append((r2 - r1) * (z0 - z1) - (z2 - z1) * (r0 - r1))
    return np.minimum.reduce(areas)


def test_inverse_harmonic_circles():
    # For concentric circles the inverse harmonic map is known exactly:
    # r = r_in * (r_out / r_in)^x, with equally spaced angles
    r_in, r_out = 0.5, 1.5
    inner = rzline.circle(R0=3.0, r=r_in, n=400)
    outer = rzline.circle(R0=3.0, r=r_out, n=400)
    nx, nz = 17, 32
    R, Z = poloidal_grid.grid_inverse_harmonic(
        inner, outer, nx, nz, return_coords=True, homogenize=0.0
    )
    x = np.linspace(0, 1, nx)[:, None]
    theta = np.linspace(0, 2 * np.pi, nz, endpoint=False)[None, :]
    r = r_in * (r_out / r_in) ** x
    assert np.allclose(R, 3.0 + r * np.cos(theta), atol=1e-6)
    assert np.allclose(Z, r * np.sin(theta), atol=1e-6)


def test_inverse_harmonic_boundaries():
    inner = rzline.shaped_line(R0=3.0, a=0.4, elong=0.3, triang=0.2, n=100)
    outer = lobed_line()
    R, Z = poloidal_grid.grid_inverse_harmonic(inner, outer, 12, 64, return_coords=True)
    # Boundary points lie on the boundary lines, here sampled densely
    for line, r, z in ((inner, R[0], Z[0]), (outer, R[-1], Z[-1])):
        theta = np.linspace(0, 2 * np.pi, 200000, endpoint=False)
        ring = shapely.LinearRing(np.column_stack(line.position(theta)))
        distance = shapely.distance(ring, shapely.points(np.column_stack([r, z])))
        assert np.amax(distance) < 1e-8


def fd_jacobian(R, Z):
    """Jacobian of the grid from centred differences in index space"""
    Rx, Zx = np.gradient(R, axis=0), np.gradient(Z, axis=0)
    Rz = 0.5 * (np.roll(R, -1, axis=1) - np.roll(R, 1, axis=1))
    Zz = 0.5 * (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1))
    return Rx * Zz - Rz * Zx


def test_inverse_harmonic_no_folding():
    # Strongly shaped outer boundary: all cells are convex
    inner = rzline.circle(R0=3.0, r=0.3, n=100)
    outer = lobed_line(amp=0.45)
    R, Z = poloidal_grid.grid_inverse_harmonic(inner, outer, 24, 64, return_coords=True)
    assert np.all(cell_areas(R, Z) > 0)
    assert np.all(fd_jacobian(R, Z) > 0)


def test_inverse_harmonic_extreme_shape():
    # Deep concave regions close to an off-centre inner boundary. The
    # spline map is valid (otherwise an exception is raised) and so is
    # the finite difference Jacobian, but the grid is so strongly sheared
    # near the concave regions that a few quadrilaterals are not convex.
    inner = rzline.circle(R0=3.2, r=0.3, n=100)
    outer = lobed_line()
    for homogenize in (0.0, 1.0):
        with pytest.warns(
            UserWarning, match="between the boundaries are folded or not"
        ):
            R, Z = poloidal_grid.grid_inverse_harmonic(
                inner, outer, 24, 64, return_coords=True, homogenize=homogenize
            )
        J = fd_jacobian(R, Z)
        assert np.amin(J) > 0.05 * np.mean(J)
        # Only a few cells next to the outer boundary are affected
        bad = np.argwhere(cell_areas(R, Z) <= 0)
        assert 0 < len(bad) <= 10 and np.all(bad[:, 0] >= 18)


def test_inverse_harmonic_homogenize():
    inner = rzline.circle(R0=3.0, r=0.3, n=100)
    outer = lobed_line()
    spread = []
    for homogenize in (0.0, 1.0):
        R, Z = poloidal_grid.grid_inverse_harmonic(
            inner, outer, 16, 64, return_coords=True, homogenize=homogenize
        )
        areas = np.abs(cell_areas(R, Z))
        spread.append(areas.max() / areas.min())
    assert spread[1] < spread[0]


def test_inverse_harmonic_extra_points():
    inner = rzline.circle(R0=3.0, r=0.5, n=100)
    outer = rzline.shaped_line(R0=3.0, a=1.2, elong=0.4, triang=0.3, n=100)
    nx, nz = 16, 32
    R0, Z0 = poloidal_grid.grid_inverse_harmonic(
        inner, outer, nx, nz, return_coords=True
    )
    R, Z = poloidal_grid.grid_inverse_harmonic(
        inner, outer, nx + 3, nz, nx_inner=1, nx_outer=2, return_coords=True
    )
    assert R.shape == (nx + 3, nz)
    # The points between the boundaries are unchanged
    assert np.allclose(R[1:-2], R0) and np.allclose(Z[1:-2], Z0)
    # The extrapolated points continue the grid lines without folding
    assert np.all(cell_areas(R, Z) > 0)


def test_inverse_harmonic_grid():
    inner = rzline.circle(R0=0.0, r=1.0, n=20)
    outer = rzline.circle(R0=0.0, r=2.0, n=20)
    grid = poloidal_grid.grid_inverse_harmonic(inner, outer, 10, 16)
    assert isinstance(grid, poloidal_grid.StructuredPoloidalGrid)
    x, y = grid.findIndex(1.0, 1.0)
    r, z = grid.getCoordinate(x, y)
    assert np.allclose(r, 1.0) and np.allclose(z, 1.0)


def metric_second_differences(R, Z):
    """Second differences of the index space metric, relative to the local
    metric magnitude, excluding the two outermost rows on each side"""
    Rx, Zx = np.gradient(R, axis=0), np.gradient(Z, axis=0)
    Rz = 0.5 * (np.roll(R, -1, axis=1) - np.roll(R, 1, axis=1))
    Zz = 0.5 * (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1))
    gxx, gxz, gzz = (g[2:-2] for g in (Rx**2 + Zx**2, Rx * Rz + Zx * Zz, Rz**2 + Zz**2))
    out = []
    for g, scale in ((gxx, gxx), (gzz, gzz), (gxz, np.sqrt(gxx * gzz))):
        out.append(np.abs(g[2:] - 2 * g[1:-1] + g[:-2]) / scale[1:-1])
        out.append(np.abs(np.roll(g, -1, 1) - 2 * g + np.roll(g, 1, 1)) / scale)
    return np.concatenate([d.ravel() for d in out])


def test_inverse_harmonic_smooth_metric():
    # Strongly triangular D shape. The metric is smooth almost everywhere;
    # close to the sharp tips of the outer boundary it changes quickly over
    # the last few cells, which limits the largest second difference.
    inner = rzline.shaped_line(R0=3.0, a=0.3, n=100)
    outer = rzline.shaped_line(R0=3.0, a=1.2, elong=1.0, triang=0.9, n=300)
    previous = None
    for nx, nz in ((17, 64), (33, 128)):
        R, Z = poloidal_grid.grid_inverse_harmonic(
            inner, outer, nx, nz, return_coords=True
        )
        d2 = metric_second_differences(R, Z)
        assert np.percentile(d2, 99) < 0.15 and np.amax(d2) < 0.5
        if previous is not None:
            # Smooth metric: second differences decrease under refinement
            assert np.percentile(d2, 99) < 0.5 * previous
        previous = np.percentile(d2, 99)


def test_inverse_harmonic_invalid_input():
    small = rzline.circle(R0=3.0, r=0.5, n=50)
    large = rzline.circle(R0=3.0, r=1.5, n=50)
    with pytest.raises(ValueError, match="inside the outer"):
        poloidal_grid.grid_inverse_harmonic(large, small, 10, 16)
    with pytest.raises(ValueError, match="inside the outer"):
        poloidal_grid.grid_inverse_harmonic(rzline.circle(R0=4.2, r=0.5), large, 10, 16)
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    figure_eight = rzline.RZline(
        3.0 + 0.5 * np.sin(theta), 0.5 * np.sin(2 * theta), anticlockwise=False
    )
    with pytest.raises(ValueError, match="intersects itself"):
        poloidal_grid.grid_inverse_harmonic(figure_eight, large, 10, 16)
    with pytest.raises(ValueError, match="resolution"):
        poloidal_grid.grid_inverse_harmonic(small, large, 10, 16, resolution=1.5)


def test_inverse_harmonic_like_elliptic():
    inner = rzline.circle(R0=3.0, r=0.5, n=50)
    outer = rzline.shaped_line(R0=3.0, a=1.2, elong=0.4, triang=0.3, n=100)
    R0, Z0 = poloidal_grid.grid_inverse_harmonic(
        inner, outer, 10, 16, return_coords=True
    )
    # Same positional arguments and conventions as grid_elliptic
    grid = poloidal_grid.grid_inverse_harmonic(inner, outer, 10, 16, False, 1e-10, True)
    assert np.allclose(grid.R, R0) and np.allclose(grid.Z, Z0)
    Re, Ze = poloidal_grid.grid_elliptic(inner, outer, 10, 16, return_coords=True)
    # Outer boundary starts at the same point; index 0 is the inner boundary
    assert np.allclose([R0[-1, 0], Z0[-1, 0]], [Re[-1, 0], Ze[-1, 0]])
    assert np.sign(fd_jacobian(R0, Z0).mean()) == np.sign(fd_jacobian(Re, Ze).mean())
    # Options of grid_elliptic are accepted but ignored
    with pytest.warns(UserWarning, match="ignoring grid_elliptic options"):
        R, Z = poloidal_grid.grid_inverse_harmonic(
            inner, outer, 10, 16, return_coords=True, inner_ort=True, restrict_size=20
        )
    assert np.allclose(R, R0) and np.allclose(Z, Z0)
    with pytest.raises(TypeError):
        poloidal_grid.grid_inverse_harmonic(inner, outer, 10, 16, unknown_option=1)


def test_inverse_harmonic_clockwise():
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    r = 1.2 * (1 + 0.2 * np.cos(3 * theta))
    R, Z = 3.0 + r * np.cos(theta), r * np.sin(theta)
    inner = rzline.circle(R0=3.0, r=0.4, n=50)
    expected = poloidal_grid.grid_inverse_harmonic(
        inner, rzline.RZline(R, Z), 10, 32, return_coords=True
    )
    # The same line, starting at the same point but going clockwise
    clockwise = rzline.RZline(
        np.roll(R[::-1], 1), np.roll(Z[::-1], 1), anticlockwise=False
    )
    result = poloidal_grid.grid_inverse_harmonic(
        inner, clockwise, 10, 32, return_coords=True
    )
    assert np.allclose(result, expected, atol=1e-8)


def test_inverse_harmonic_linear_solvers(monkeypatch):
    inner = rzline.circle(R0=3.0, r=0.4, n=50)
    outer = lobed_line(amp=0.3)
    superlu = poloidal_grid.grid_inverse_harmonic(
        inner, outer, 12, 32, return_coords=True, linear_solver="superlu"
    )
    if poloidal_grid.pypardiso is not None:
        pardiso = poloidal_grid.grid_inverse_harmonic(
            inner, outer, 12, 32, return_coords=True, linear_solver="pardiso"
        )
        assert np.allclose(pardiso, superlu, atol=1e-10)
    # Without the optional pypardiso, "auto" falls back to SuperLU
    monkeypatch.setattr(poloidal_grid, "pypardiso", None)
    auto = poloidal_grid.grid_inverse_harmonic(inner, outer, 12, 32, return_coords=True)
    assert np.allclose(auto, superlu, atol=1e-10)
    with pytest.raises(ImportError):
        poloidal_grid.grid_inverse_harmonic(
            inner, outer, 12, 32, linear_solver="pardiso"
        )


def slotted_circle(R0=3.0, r=1.5, width=0.05, depth=0.9, lean=0.3, n=600):
    """Circle with a narrow slot cut in from the bottom (closed divertor)"""
    theta = np.linspace(
        -np.pi / 2 + width / (2 * r), 3 * np.pi / 2 - width / (2 * r), n
    )
    points = list(zip(R0 + r * np.cos(theta), r * np.sin(theta)))
    d = np.array([np.sin(lean), np.cos(lean)])  # Into the domain
    normal = np.array([d[1], -d[0]])
    mouth = np.array([R0, -np.sqrt(r**2 - (width / 2) ** 2)])
    side = np.linspace(0, depth, int(depth / 0.02))
    points += [mouth - normal * width / 2 + s * d for s in side[1:]]
    angle = np.linspace(np.pi, 0, 12)[1:-1]
    tip = mouth + depth * d
    points += [tip + width / 2 * (np.cos(a) * normal + np.sin(a) * d) for a in angle]
    points += [mouth + normal * width / 2 + s * d for s in side[::-1][:-1]]
    points = np.array(points)
    return rzline.RZline(points[:, 0], points[:, 1])


def signed_cell_areas(R, Z):
    R = np.concatenate([R, R[:, :1]], axis=1)
    Z = np.concatenate([Z, Z[:, :1]], axis=1)
    c = [(R[:-1, :-1], Z[:-1, :-1]), (R[1:, :-1], Z[1:, :-1])]
    c += [(R[1:, 1:], Z[1:, 1:]), (R[:-1, 1:], Z[:-1, 1:])]
    return 0.5 * sum(c[i][0] * c[i - 3][1] - c[i - 3][0] * c[i][1] for i in range(4))


def test_inverse_harmonic_continuation():
    # A narrow, leaning slot: no valid grid is found directly, but with
    # the continuation from a smoothed outer line
    inner = rzline.circle(R0=3.0, r=0.4, n=100)
    outer = slotted_circle()
    kw = {"homogenize": 0.0, "max_refine": 0}
    with pytest.raises(RuntimeError):
        poloidal_grid.grid_inverse_harmonic(
            inner, outer, 12, 64, continuation=False, **kw
        )
    R, Z = poloidal_grid.grid_inverse_harmonic(
        inner, outer, 12, 64, return_coords=True, **kw
    )
    assert np.all(signed_cell_areas(R, Z) > 0)
    # The grid fits the given outer line, not the smoothed one
    theta = np.linspace(0, 2 * np.pi, 200000, endpoint=False)
    ring = shapely.LinearRing(np.column_stack(outer.position(theta)))
    distance = shapely.distance(ring, shapely.points(np.column_stack([R[-1], Z[-1]])))
    assert np.amax(distance) < 1e-6


def test_inverse_harmonic_curvature_weight():
    # Elongated ellipse: weighting by curvature puts more points at the tips
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    outer = rzline.RZline(3.0 + 1.5 * np.cos(theta), 0.5 * np.sin(theta))
    inner = rzline.circle(R0=3.0, r=0.2, n=100)
    ratio = []
    for weight in (0.0, 0.5):
        R, Z = poloidal_grid.grid_inverse_harmonic(
            inner, outer, 8, 64, return_coords=True, curvature_weight=weight
        )
        spacing = np.hypot(
            np.diff(R[-1], append=R[-1, 0]), np.diff(Z[-1], append=Z[-1, 0])
        )
        tips = np.abs(R[-1] - 3.0) > 1.3
        ratio.append(spacing[tips].mean() / spacing[~tips].mean())
    assert ratio[1] < 0.6 * ratio[0]
