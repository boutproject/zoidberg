"""Tests of the triangulated wall

The walls are built from analytic families of contours C(phi), periodic
with yperiod, so that the mesh can be compared with the exact surface
between the cross-sections it was built from.
"""

import numpy as np
import pytest

from . import rzline, wallgeometry

shapely = pytest.importorskip("shapely")

yperiod = 2 * np.pi / 5


def ellipse(phi, t):
    """Ellipse rotating by half a turn per period, t in [0, 1)"""
    alpha = np.pi * phi / yperiod
    x, y = 0.6 * np.cos(2 * np.pi * t), 0.3 * np.sin(2 * np.pi * t)
    return (
        2 + x * np.cos(alpha) - y * np.sin(alpha),
        x * np.sin(alpha) + y * np.cos(alpha),
    )


def bean(phi, t):
    """Concave, changing shape whose centre moves"""
    th = 2 * np.pi * t
    w = 2 * np.pi * phi / yperiod
    rho = 0.5 + 0.15 * np.cos(2 * th - w) + 0.12 * np.cos(3 * th + w)
    return (
        2 + 0.1 * np.cos(w) + rho * np.cos(th),
        0.1 * np.sin(w) + rho * np.sin(th),
    )


def roaming(phi, t):
    """The centre moves by more than the size of the contour, whose size
    changes by a factor of six"""
    w = 2 * np.pi * phi / yperiod
    th = 2 * np.pi * t
    rho = (0.35 + 0.25 * np.sin(w)) * (1 + 0.3 * np.cos(2 * th + w))
    return 2 + 0.6 * np.cos(w) + rho * np.cos(th), 0.6 * np.sin(w) + rho * np.sin(th)


def notch_polygon(phi, narc=20):
    """D-shape with a divertor notch that grows from nothing, with sharp
    corners and only the corners on the straight parts"""
    depth = 0.2 * (1 - np.cos(2 * np.pi * phi / yperiod))
    arc = np.linspace(-0.45 * np.pi, 0.45 * np.pi, narc)
    P = np.vstack(
        (
            np.column_stack((0.55 * np.cos(arc), 0.7 * np.sin(arc))),
            [[-0.45, 0.62], [-0.5, 0.0], [-0.45, -0.62], [-0.25, -0.62]],
            [[-0.15, -0.62 + depth], [0.0, -0.62 + depth], [0.08, -0.62]],
        )
    )
    return P + [2.0, 0.0]


def notch(phi, t):
    P = notch_polygon(phi, narc=400)
    P = np.vstack((P, P[:1]))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate(([0], np.cumsum(seg))) / seg.sum()
    return np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])


def sample(func, phis, npoints, seed=0, random_start=False, warp=False):
    """Contours of func at phis. npoints(j) points in section j"""
    rng = np.random.default_rng(seed)
    lines = []
    for j, phi in enumerate(phis):
        n = npoints(j)
        t = np.arange(n) / n
        if warp:
            # Points cluster in a region that moves with phi
            t = t + 0.08 * np.sin(2 * np.pi * (t - phi / yperiod))
        if random_start:
            t = np.roll(t, rng.integers(n))
        lines.append(rzline.RZline(*func(phi, t), anticlockwise=False))
    return lines


def notch_lines(phis):
    lines = []
    for j, phi in enumerate(phis):
        P = notch_polygon(phi, narc=20 + (7 * j) % 40)
        lines.append(rzline.RZline(P[:, 0], P[:, 1], anticlockwise=False))
    return lines


def phis_for(nsections):
    return np.linspace(0, yperiod, nsections, endpoint=False)


def mid_phis(wall):
    return 0.5 * (wall.mesh_phi[:-1] + wall.mesh_phi[1:])


def error_to_exact(wall, func, every=1):
    """Largest Hausdorff distance and relative area error between the
    mesh and the exact surface, in the middle of every strip (or of
    every `every`-th strip)"""
    t = np.linspace(0, 1, 4000, endpoint=False)
    distance, area = 0.0, 0.0
    for phi in mid_phis(wall)[::every]:
        R, Z = wall.section(phi)
        ring = shapely.LinearRing(np.column_stack((R, Z)))
        exact = shapely.LinearRing(np.column_stack(func(phi, t)))
        distance = max(distance, curve_distance(np.column_stack((R, Z)), exact))
        exact_area = shapely.Polygon(exact).area
        area = max(area, abs(shapely.Polygon(ring).area - exact_area) / exact_area)
    return distance, area


def curve_distance(points, exact):
    """Hausdorff distance between the closed polygon through points and
    a densely sampled exact curve (a LinearRing)"""
    mid = 0.5 * (points + np.roll(points, -1, axis=0))
    polygon = shapely.LinearRing(points)
    exact_points = shapely.get_coordinates(exact)
    return max(
        np.max(shapely.distance(shapely.points(np.vstack((points, mid))), exact)),
        np.max(shapely.distance(shapely.points(exact_points), polygon)),
    )


def edge_counts(triangles):
    edges = np.sort(
        np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        ),
        axis=1,
    )
    return np.unique(edges, axis=0, return_counts=True)


def brute_force_dp(qa, qb):
    """Minimal path cost, written as plainly as possible"""
    m, n = len(qa) - 1, len(qb) - 1
    cost = np.array(
        [[np.sum((qa[i] - qb[j]) ** 2) for j in range(n + 1)] for i in range(m + 1)]
    )
    D = np.full((m + 1, n + 1), np.inf)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                D[i, j] = cost[i, j]
            else:
                up = D[i - 1, j] if i > 0 else np.inf
                left = D[i, j - 1] if j > 0 else np.inf
                D[i, j] = cost[i, j] + min(up, left)
    return D[m, n]


# ---------------------------------------------------------------------
# Input contours


def test_clean_contour_orients_anticlockwise():
    t = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    for sign in (1, -1):
        P = wallgeometry.clean_contour(2 + np.cos(sign * t), np.sin(sign * t))
        assert wallgeometry._signed_area(P[:, 0], P[:, 1]) > 0
        assert len(P) == 10


def test_clean_contour_removes_repeated_points():
    R = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0])
    Z = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])  # Last point closes the polygon
    P = wallgeometry.clean_contour(R, Z)
    assert len(P) == 4
    assert len(np.unique(P, axis=0)) == 4


@pytest.mark.parametrize(
    "R, Z, message",
    [
        ([1.0, 2.0], [0.0, 1.0], "three distinct"),
        ([1.0, 2.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], "three distinct"),
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], "no area"),
        ([1.0, 3.0, 3.0, 1.0], [0.0, 2.0, 0.0, 1.0], "intersects itself"),
        ([1.0, 2.0, np.nan], [0.0, 0.0, 1.0], "non-finite"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "no area"),
    ],
)
def test_clean_contour_rejects_invalid(R, Z, message):
    with pytest.raises(ValueError, match=message):
        wallgeometry.clean_contour(np.array(R), np.array(Z))


def test_wall_reports_invalid_section():
    lines = sample(ellipse, phis_for(4), lambda j: 16)
    lines[2] = rzline.RZline(
        np.array([1.0, 2.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        anticlockwise=False,
    )
    with pytest.raises(ValueError, match="cross-section 2"):
        wallgeometry.Wall(lines)


@pytest.mark.parametrize(
    "phi, kwargs, message",
    [
        (phis_for(4)[:3], {"yperiod": yperiod}, "one angle per"),
        (phis_for(4)[::-1], {"yperiod": yperiod}, "increasing"),
        (np.linspace(0, yperiod, 4), {"yperiod": yperiod}, "within one period"),
        (np.linspace(0, 2 * np.pi, 4), {"close_toroidal": True}, "within one period"),
    ],
)
def test_create_mesh_rejects_invalid_angles(phi, kwargs, message):
    wall = wallgeometry.Wall(sample(ellipse, phis_for(4), lambda j: 16))
    with pytest.raises(ValueError, match=message):
        wall.create_mesh(phi, **kwargs)


def test_open_wall_needs_two_sections():
    wall = wallgeometry.Wall(sample(ellipse, [0.0], lambda j: 16))
    with pytest.raises(ValueError, match="two cross-sections"):
        wall.create_mesh([0.0])
    # A single section is fine for a periodic wall
    wall.create_mesh([0.0], yperiod=0.1)
    assert len(wall.triangles) == 32


# ---------------------------------------------------------------------
# Correspondence between contours


def test_dp_batch_is_optimal():
    """The batched, padded dynamic programming finds the minimal cost
    path, for problems of different sizes in one batch"""
    rng = np.random.default_rng(3)
    problems = []
    for m, n in [(3, 3), (3, 9), (8, 4), (12, 12), (5, 17)]:
        qa = rng.normal(size=(m, 2))
        qb = rng.normal(size=(n, 2))
        problems.append((np.vstack((qa, qa[:1])), np.vstack((qb, qb[:1]))))
    paths, costs = wallgeometry._dp_batch(problems)
    for (qa, qb), path, total in zip(problems, paths, costs):
        m, n = len(qa) - 1, len(qb) - 1
        assert total == pytest.approx(brute_force_dp(qa, qb), rel=1e-12)
        # The path is monotone, from (0, 0) to (m, n), one step at a time
        assert len(path) == m + n + 1
        assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (m, n)
        steps = np.diff(path, axis=0)
        assert np.all(np.sum(steps, axis=1) == 1) and np.all(steps >= 0)
        # and its cost is the reported optimum
        path_cost = np.sum((qa[path[:, 0]] - qb[path[:, 1]]) ** 2)
        assert path_cost == pytest.approx(total, rel=1e-12)


def test_batching_does_not_change_the_mesh(monkeypatch):
    lines = sample(bean, phis_for(12), lambda j: 20 + 3 * j, random_start=True)
    reference = wallgeometry.Wall(lines, phi=phis_for(12), yperiod=yperiod)
    # One strip per batch
    monkeypatch.setattr(wallgeometry, "_DP_BATCH_CELLS", 1)
    wall = wallgeometry.Wall(lines, phi=phis_for(12), yperiod=yperiod)
    assert np.array_equal(wall.triangles, reference.triangles)


def test_start_point_finds_shift():
    """The best start point of B, for B a rolled, shifted and scaled copy of A"""
    t = np.linspace(0, 1, 50, endpoint=False)
    A = wallgeometry.clean_contour(*bean(0.0, t))
    qa, sa = wallgeometry._normalise(A)
    for roll in (0, 7, 31):
        B = 3.0 * np.roll(A, -roll, axis=0) + [5.0, -2.0]
        qb, sb = wallgeometry._normalise(B)
        assert wallgeometry._start_point(qa, sa, qb, sb) == (-roll) % 50


def test_rungs_cover_both_contours():
    lines = sample(bean, phis_for(2), lambda j: (40, 73)[j], random_start=True)
    contours = wallgeometry.Wall(lines).contours
    (rungs,) = wallgeometry.tile_contours(contours, [(0, 1)])
    assert len(rungs) == 40 + 73 + 1
    assert tuple(rungs[0]) == tuple(rungs[-1])
    di = (np.diff(rungs[:, 0]) % 40) == 1
    dj = (np.diff(rungs[:, 1]) % 73) == 1
    # Every step advances exactly one contour by one point
    assert np.all(di ^ dj)
    assert di.sum() == 40 and dj.sum() == 73


# ---------------------------------------------------------------------
# Mesh


CASES = {
    "ellipse": (ellipse, 64, {"npoints": lambda j: 64}),
    "ellipse_random_start": (
        ellipse,
        64,
        {"npoints": lambda j: 64, "random_start": True},
    ),
    "bean_clustered": (bean, 48, {"npoints": lambda j: 96, "warp": True}),
    "bean_varying_points": (
        bean,
        48,
        {"npoints": lambda j: 60 + (37 * j) % 80, "warp": True, "random_start": True},
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_mesh_close_to_exact_surface(case):
    func, nsections, kwargs = CASES[case]
    phis = phis_for(nsections)
    wall = wallgeometry.Wall(sample(func, phis, **kwargs), phi=phis, yperiod=yperiod)
    distance, area = error_to_exact(wall, func)
    assert distance < 0.006
    assert area < 0.005


def test_closure_of_rotating_contours():
    """Regression: the start points of the contours turn by half a turn
    over the period. Aligning neighbours only closed the period through
    the inside of the wall, so that the wall was pinched there"""
    phis = phis_for(64)
    wall = wallgeometry.Wall(
        sample(ellipse, phis, lambda j: 64), phi=phis, yperiod=yperiod
    )
    phi = 0.5 * (phis[-1] + yperiod)
    R, Z = wall.section(phi)
    assert abs(wallgeometry._signed_area(R, Z)) == pytest.approx(
        np.pi * 0.18, rel=0.005
    )
    # The rungs of the closing strip are as short as those of the others
    longest = [
        np.max(np.linalg.norm(wall.vertices[r[:, 0]] - wall.vertices[r[:, 1]], axis=1))
        for r in wall.strip_rungs
    ]
    assert longest[-1] <= 1.01 * max(longest[:-1])


def test_contours_moving_and_changing_size():
    """Matching is done in coordinates normalised by the centroid and
    the size of the contours. In R, Z the mesh would fold over here."""
    phis = phis_for(32)
    lines = sample(roaming, phis, lambda j: 90 + 7 * j, random_start=True)
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    distance, area = error_to_exact(wall, roaming)
    assert distance < 0.01
    assert area < 0.04


def test_sharp_corners_and_growing_notch():
    for nsections, tolerance in ((40, 0.008), (160, 0.003)):
        phis = phis_for(nsections)
        wall = wallgeometry.Wall(notch_lines(phis), phi=phis, yperiod=yperiod)
        distance, _ = error_to_exact(wall, notch)
        assert distance < tolerance


def test_second_order_convergence_with_sections():
    """The faceted wall between the sections converges with second order"""
    levels = np.array([16, 32, 64, 128])
    errors = []
    for nsections in levels:
        phis = phis_for(nsections)
        lines = sample(ellipse, phis, lambda j: 512 + (j % 3), random_start=True)
        wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
        # The ellipse turns rigidly, so a few strips are representative
        errors.append(error_to_exact(wall, ellipse, every=nsections // 8)[0])
    order = -np.polyfit(np.log(levels), np.log(errors), 1)[0]
    assert order == pytest.approx(2, abs=0.25), errors


def test_input_points_are_vertices():
    phis = phis_for(8)
    lines = notch_lines(phis)
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    for line, phi in zip(lines, phis):
        xyz = np.column_stack(
            wallgeometry.rzphi_to_xyz(line.R, line.Z, np.full(line.R.size, phi))
        )
        distance = np.min(
            np.linalg.norm(xyz[:, None] - wall.vertices[None], axis=-1), axis=1
        )
        assert np.all(distance < 1e-12)


def test_section_at_input_angles_is_the_contour():
    phis = phis_for(8)
    lines = sample(bean, phis, lambda j: 30 + j, random_start=True)
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    for k, phi in enumerate(np.append(phis, yperiod)):
        R, Z = wall.section(phi)
        contour = wall.contours[k % len(phis)]
        ring = shapely.LinearRing(contour)
        assert (
            shapely.hausdorff_distance(
                shapely.LinearRing(np.column_stack((R, Z))), ring
            )
            < 1e-12
        )
        assert wallgeometry._signed_area(R, Z) > 0


def test_section_is_periodic():
    phis = phis_for(8)
    wall = wallgeometry.Wall(
        sample(bean, phis, lambda j: 30 + j), phi=phis, yperiod=yperiod
    )
    R, Z = wall.section(0.3)
    for k in (-3, 1, 7):
        Rk, Zk = wall.section(0.3 + k * yperiod)
        assert np.allclose(Rk, R, atol=1e-12) and np.allclose(Zk, Z, atol=1e-12)


@pytest.mark.parametrize("case", ["bean_varying_points", "notch"])
def test_normals_point_out_of_concave_walls(case):
    """A small step along the normal leaves the wall, against it enters"""
    phis = phis_for(24)
    if case == "notch":
        lines = notch_lines(phis)
    else:
        lines = sample(bean, phis, CASES[case][2]["npoints"], random_start=True)
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    tri = wall.vertices[wall.triangles]
    centroid = tri.mean(axis=1)
    rng = np.random.default_rng(0)
    for k in rng.choice(len(tri), 300, replace=False):
        for sign, inside in ((1, False), (-1, True)):
            p = centroid[k] + sign * 1e-5 * wall.normals[k]
            phi = np.arctan2(p[1], p[0]) % (2 * np.pi)
            R, Z = wall.section(phi)
            polygon = shapely.Polygon(np.column_stack((R, Z)))
            assert polygon.contains(shapely.Point(np.hypot(p[0], p[1]), p[2])) == inside


def test_independent_of_start_point_and_orientation():
    phis = phis_for(16)
    reference = wallgeometry.Wall(
        sample(bean, phis, lambda j: 64), phi=phis, yperiod=yperiod
    )
    lines = sample(bean, phis, lambda j: 64, random_start=True)
    # Reverse every other contour
    for line in lines[::2]:
        line.R, line.Z = line.R[::-1].copy(), line.Z[::-1].copy()
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    for phi in mid_phis(wall):
        a = shapely.LinearRing(np.column_stack(wall.section(phi)))
        b = shapely.LinearRing(np.column_stack(reference.section(phi)))
        assert shapely.hausdorff_distance(a, b) < 1e-12


@pytest.mark.parametrize("periodic", ["yperiod", "close_toroidal", "open"])
def test_mesh_topology(periodic):
    """Every edge has two triangles, except on the end sections of the mesh.
    The mesh is a tube: V - E + F = 0"""
    if periodic == "close_toroidal":
        phis = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        kwargs = {"close_toroidal": True}
    else:
        phis = phis_for(20)
        kwargs = {"yperiod": yperiod} if periodic == "yperiod" else {}
    lines = sample(bean, phis, lambda j: 30 + (5 * j) % 17, random_start=True)
    wall = wallgeometry.Wall(lines, phi=phis, **kwargs)
    edges, count = edge_counts(wall.triangles)
    assert set(count) == {1, 2}
    n_first = len(wall.contours[0])
    n_last = n_first if periodic != "open" else len(wall.contours[-1])
    nv = len(wall.vertices)
    single = edges[count == 1]
    assert np.all(
        np.all(single < n_first, axis=1) | np.all(single >= nv - n_last, axis=1)
    )
    assert np.sum(count == 1) == n_first + n_last
    assert nv - len(edges) + len(wall.triangles) == 0
    if periodic == "close_toroidal":
        assert wall.yperiod == 2 * np.pi


def test_create_wall_mesh_needs_one_angle_per_section():
    contours = wallgeometry.Wall(sample(bean, phis_for(3), lambda j: 20)).contours
    with pytest.raises(ValueError, match="one angle per"):
        wallgeometry.create_wall_mesh(contours, phis_for(4))


def test_create_wall_mesh_closed_torus():
    """Without yperiod, close_toroidal joins the last section to the first"""
    phis = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    contours = wallgeometry.Wall(sample(bean, phis, lambda j: 20 + j)).contours
    vertices, triangles, rungs = wallgeometry.create_wall_mesh(
        contours, phis, close_toroidal=True
    )
    _, count = edge_counts(triangles)
    assert np.all(count == 2)
    assert len(rungs) == 12
    assert len(vertices) == sum(len(c) for c in contours)


def test_open_wall():
    phis = np.linspace(0.1, 0.9, 10)
    wall = wallgeometry.Wall(sample(bean, phis, lambda j: 40), phi=phis)
    assert wall.yperiod is None
    assert np.array_equal(wall.mesh_phi, phis)
    wall.section(0.5)
    wall.section(0.9)
    for phi in (0.05, 0.95):
        with pytest.raises(ValueError, match="outside the wall"):
            wall.section(phi)


def test_mesh_is_deterministic():
    phis = phis_for(16)
    lines = sample(bean, phis, lambda j: 40 + j, random_start=True)
    a = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    b = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    assert np.array_equal(a.triangles, b.triangles)
    assert np.array_equal(a.vertices, b.vertices)


# ---------------------------------------------------------------------
# Mesh checks


@pytest.fixture
def good_wall():
    phis = phis_for(16)
    return wallgeometry.Wall(
        sample(bean, phis, lambda j: 40 + j, random_start=True),
        phi=phis,
        yperiod=yperiod,
    )


def rebuild_triangles(wall):
    """Triangles from modified rungs"""
    wall.triangles = np.vstack(
        [
            wallgeometry._strip_triangles(
                np.append(r[:, 0], r[0, 0]), np.append(r[:, 1], r[0, 1])
            )
            for r in wall.strip_rungs
        ]
    )


def test_check_passes(good_wall):
    good_wall.check()


def test_check_detects_twisted_strip(good_wall):
    # Join the closing strip half way round, as the old alignment did
    rungs = good_wall.strip_rungs[-1]
    n = len(good_wall.contours[0])
    first = len(good_wall.vertices) - n
    rungs[:, 1] = first + (rungs[:, 1] - first + n // 2) % n
    rebuild_triangles(good_wall)
    with pytest.raises(ValueError, match="(pinched|folds)"):
        good_wall.check()


def test_check_detects_invalid_rungs(good_wall):
    # Points of section 4 skipped, then visited backwards
    rungs = good_wall.strip_rungs[3]
    first = sum(len(c) for c in good_wall.contours[:4])
    n = len(good_wall.contours[4])
    rungs[40:46, 1] = first + (rungs[40:46, 1] - first + 3) % n
    with pytest.raises(
        ValueError, match="Invalid rungs between cross-sections 3 and 4"
    ):
        good_wall.check()


def test_check_detects_triangles_not_matching_rungs(good_wall):
    good_wall.triangles = good_wall.triangles[:, [1, 2, 0]]
    with pytest.raises(ValueError, match="do not match"):
        good_wall.check()


def test_check_detects_fold(monkeypatch):
    """Matched in R, Z instead of normalised coordinates, the wall of
    contours that move and change size folds over"""
    normalise = wallgeometry._normalise
    monkeypatch.setattr(
        wallgeometry, "_normalise", lambda P: (P.copy(), normalise(P)[1])
    )
    phis = phis_for(16)
    lines = sample(roaming, phis, lambda j: 90 + 7 * j, random_start=True)
    with pytest.raises(ValueError, match="folds over"):
        wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)


def test_check_detects_pinch(good_wall):
    # The sections of the good wall do not shrink between the
    # cross-sections, so a threshold above one flags every strip
    good_wall.check(min_area_ratio=0.95)
    with pytest.raises(ValueError, match="pinched between cross-sections 0 and 1"):
        good_wall.check(min_area_ratio=1.5)


def test_check_detects_cone(good_wall):
    """A strip whose rungs join all points of one contour to one point
    of the next, and back, is a valid triangulation, but not a surface"""
    k = 3
    first_a = sum(len(c) for c in good_wall.contours[:k])
    first_b = first_a + len(good_wall.contours[k])
    m, n = len(good_wall.contours[k]), len(good_wall.contours[k + 1])
    a = np.concatenate((np.arange(m), np.zeros(n, dtype=int))) + first_a
    b = np.concatenate((np.zeros(m, dtype=int), np.arange(n))) + first_b
    good_wall.strip_rungs[k] = np.column_stack((a, b))
    rebuild_triangles(good_wall)
    with pytest.raises(ValueError, match="more than two triangles"):
        good_wall.check()


def test_check_detects_degenerate_triangle(good_wall):
    a, b, _ = good_wall.triangles[10]
    good_wall.vertices[b] = good_wall.vertices[a]
    with pytest.raises(ValueError, match="degenerate"):
        good_wall.check()


def test_check_can_be_disabled(monkeypatch):
    def fail(self):
        raise AssertionError("check called")

    monkeypatch.setattr(wallgeometry.Wall, "check", fail)
    phis = phis_for(4)
    lines = sample(bean, phis, lambda j: 20)
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod, check=False)
    wall.create_mesh(phis, yperiod=yperiod, check=False)
    with pytest.raises(AssertionError, match="check called"):
        wall.create_mesh(phis, yperiod=yperiod)
