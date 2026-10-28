"""Tests of field line - wall intersections

The convergence tests use a manufactured solution: a field whose field
lines are known analytically, and a non-axisymmetric wall given by an
analytic formula. Field lines spiral around a magnetic axis at (Ra, Za)
with rotational transform iota, and drift outwards with
d(rho)/d(phi) = a:

    rho = rho0 + a * (phi - phi0)
    theta = theta0 + iota * (phi - phi0)

The wall is a circle of radius rw whose centre rotates around the axis
with N periods per turn.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from . import field, grid, intersection, poloidal_grid, rzline, wallgeometry

shapely = pytest.importorskip("shapely")

Ra, Za = 2.0, 0.0  # Magnetic axis
a, iota = 0.3, 1.5  # Radial drift, rotational transform
rw, delta, N = 0.5, 0.1, 5  # Wall radius, shift of the wall centre, periods
yperiod = 2 * np.pi / N


class SpiralField(field.MagneticField):
    """Field lines spiralling outwards around (Ra, Za)"""

    def _rho(self, x, z):
        u = x - Ra
        w = z - Za
        return u, w, np.hypot(u, w)

    def Bxfunc(self, x, z, phi):
        u, w, rho = self._rho(x, z)
        return (a * u / rho - iota * w) / x

    def Bzfunc(self, x, z, phi):
        u, w, rho = self._rho(x, z)
        return (a * w / rho + iota * u) / x

    def Byfunc(self, x, z, phi):
        return np.ones_like(x)

    def Rfunc(self, x, z, phi):
        return x


def exact_position(R0, Z0, phi0, phi):
    rho0 = np.hypot(R0 - Ra, Z0 - Za)
    theta0 = np.arctan2(Z0 - Za, R0 - Ra)
    rho = rho0 + a * (phi - phi0)
    theta = theta0 + iota * (phi - phi0)
    return Ra + rho * np.cos(theta), Za + rho * np.sin(theta)


def exact_length(R0, Z0, phi0, phi):
    rho0 = np.hypot(R0 - Ra, Z0 - Za)
    theta0 = np.arctan2(Z0 - Za, R0 - Ra)

    def dlength(p):
        rho = rho0 + a * (p - phi0)
        R = Ra + rho * np.cos(theta0 + iota * (p - phi0))
        return np.sqrt(R**2 + a**2 + iota**2 * rho**2)

    return abs(quad(dlength, phi0, phi, epsabs=1e-14, epsrel=1e-14, limit=200)[0])


def wall_centre(phi):
    return Ra + delta * np.cos(N * phi), Za + delta * np.sin(N * phi)


def make_wall(nsections, npoints):
    """Wall with nsections cross-sections per period, npoints per section"""
    phis = np.linspace(0, yperiod, nsections, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    lines = []
    for phi in phis:
        Rc, Zc = wall_centre(phi)
        lines.append(rzline.RZline(Rc + rw * np.cos(theta), Zc + rw * np.sin(theta)))
    return wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)


def make_grid(ny=4, nx=6, nz=16):
    """Annulus of points around the axis. The slices are shifted so that
    they don't coincide with the cross-sections of the wall"""
    rho = np.linspace(0.30, 0.44, nx)
    theta = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    pol = poloidal_grid.StructuredPoloidalGrid(
        Ra + rho[:, None] * np.cos(theta), Za + rho[:, None] * np.sin(theta)
    )
    ycoords = np.linspace(0, yperiod, ny, endpoint=False) + 0.0123
    return grid.Grid(pol, ycoords, yperiod, yperiodic=True)


def legs(g):
    """All legs of a grid as (prefix, j, R0, Z0, y0, y1)"""
    for prefix, offset in (("forward", 1), ("backward", -1)):
        for j in range(g.shape[1]):
            pol, y0 = g.getPoloidalGrid(j)
            _, y1 = g.getPoloidalGrid(j + offset)
            yield prefix, j, pol.R, pol.Z, y0, y1


def first_roots(func, y0, y1, nlines, nsamples=1001):
    """First root along phi from y0 to y1 of func(phi, line), NaN if none.
    func must be vectorised, returning shape (len(phi), nlines)"""
    phis = np.linspace(y0, y1, nsamples)
    values = func(phis[:, None], slice(None))
    change = np.sign(values[1:]) != np.sign(values[:-1])
    roots = np.full(nlines, np.nan)
    for line in np.flatnonzero(np.any(change, axis=0)):
        k = np.argmax(change[:, line])
        roots[line] = brentq(
            func, phis[k], phis[k + 1], args=(line,), xtol=1e-15, rtol=1e-15
        )
    return roots


def references(g, hit_phi):
    """Exact hit angle and length of all legs with a hit well inside the leg

    hit_phi(R0, Z0, y0, y1) returns the angle of the first hit of the
    field lines starting at (R0, Z0), NaN if there is none
    """
    result = {}
    nx, _, nz = g.shape
    for prefix, j, R0, Z0, y0, y1 in legs(g):
        phis = hit_phi(R0.ravel(), Z0.ravel(), y0, y1)
        fraction = (phis - y0) / (y1 - y0)
        for line in np.flatnonzero((fraction > 0.02) & (fraction < 0.98)):
            ix, iz = np.unravel_index(line, (nx, nz))
            r0, z0 = R0[ix, iz], Z0[ix, iz]
            result[(prefix, (ix, j, iz))] = (
                phis[line],
                exact_length(r0, z0, y0, phis[line]),
            )
    return result


def errors(result, refs):
    status = np.array([result[f"{p}_status"][i] for p, i in refs])
    length = np.array([result[f"{p}_hit_length"][i] for p, i in refs])
    exact = np.array([L for _, L in refs.values()])
    return status, np.abs(length - exact)


def convergence_order(h, err):
    """Least-squares slope of log(err) against log(h)"""
    return np.polyfit(np.log(h), np.log(err), 1)[0]


@pytest.fixture(scope="module")
def mms_grid():
    return make_grid()


def test_mms_tracing_convergence(mms_grid):
    """Convergence with the toroidal step along the field line

    The reference is the exact field line crossing the same faceted
    wall, so only the error of the field line representation is
    measured: fourth order with refinement, second order without.
    """
    wall = make_wall(16, 32)
    cache = intersection._wall_tree(wall)

    def hit_on_mesh(R0, Z0, y0, y1):
        # Find the triangles from finely sampled exact field lines
        phis = np.linspace(y0, y1, 1001)
        R, Z = exact_position(R0, Z0, y0, phis[:, None])
        segment, triangle, _, shift = intersection.first_crossings(wall, R, Z, phis)

        def plane(phi, line, tri):
            """Distance of the exact field line from the plane of tri"""
            R, Z = exact_position(R0[line], Z0[line], y0, phi)
            q = phi - shift[line]
            xyz = np.array([R * np.cos(q), R * np.sin(q), Z])
            normal = np.cross(cache["edge1"][tri], cache["edge2"][tri])
            return normal @ (xyz - cache["v0"][tri])

        roots = np.full(R0.size, np.nan)
        for line in np.flatnonzero(segment >= 0):
            k = segment[line]
            roots[line] = brentq(
                plane,
                phis[k],
                phis[k + 1],
                args=(line, triangle[line]),
                xtol=1e-15,
                rtol=1e-15,
            )
        return roots

    refs = references(mms_grid, hit_on_mesh)
    assert len(refs) > 100

    dleg = yperiod / mms_grid.shape[1]
    for refine, nsteps, expected in (
        (True, [4, 8, 16], 4),
        (False, [8, 16, 32, 64], 2),
    ):
        rms = []
        for n in nsteps:
            result = intersection.find_wall_intersections(
                mms_grid,
                SpiralField(),
                wall,
                dphi_max=dleg / n,
                refine=refine,
                rtol=1e-12,
                atol=1e-13,
            )
            status, err = errors(result, refs)
            # At coarse resolution a straight segment can miss a crossing
            found = status != intersection.NO_HIT
            assert np.mean(found) > 0.98
            rms.append(np.sqrt(np.mean(err[found] ** 2)))
        order = convergence_order(dleg / np.array(nsteps), rms)
        assert order == pytest.approx(expected, abs=0.4), (refine, rms)

    # At the finest resolution every crossing is found and refined
    assert np.all(status == intersection.HIT)


def hit_on_wall(R0, Z0, y0, y1):
    """First crossing of the exact field lines with the exact wall"""

    def distance(phi, line):
        R, Z = exact_position(R0[line], Z0[line], y0, phi)
        Rc, Zc = wall_centre(phi)
        return (R - Rc) ** 2 + (Z - Zc) ** 2 - rw**2

    return first_roots(distance, y0, y1, R0.size)


def test_mms_wall_convergence(mms_grid):
    """Convergence with the resolution of the wall, against the exact
    (continuous) wall. The faceted wall converges with second order."""
    refs = references(mms_grid, hit_on_wall)
    assert len(refs) > 100

    dleg = yperiod / mms_grid.shape[1]
    levels = np.arange(1, 6)
    rms = []
    for level in levels:
        wall = make_wall(8 * 2**level, 16 * 2**level)
        result = intersection.find_wall_intersections(
            mms_grid, SpiralField(), wall, dphi_max=dleg / 32
        )
        status, err = errors(result, refs)
        assert np.all(status == intersection.HIT)
        rms.append(np.sqrt(np.mean(err**2)))

    order = convergence_order(0.5**levels, rms)
    assert order == pytest.approx(2, abs=0.3), rms

    # Angle between field line and wall at the finest resolution
    angle = angle_errors(mms_grid, result, refs)
    assert np.max(angle) < 0.02


def exact_angle(R0, Z0, y0, phi):
    """Angle between the exact field line from (R0, Z0, y0) and the exact
    wall, the surface (R - Rc(phi))^2 + (Z - Zc(phi))^2 = rw^2, at phi"""
    R, Z = exact_position(R0, Z0, y0, phi)
    u, w = R - Ra, Z - Za
    rho = np.hypot(u, w)
    # (dR/dphi, R, dZ/dphi) in (R, phi, Z) components
    tangent = np.array([a * u / rho - iota * w, R, a * w / rho + iota * u])
    Rc, Zc = wall_centre(phi)
    dRc, dZc = -delta * N * np.sin(N * phi), delta * N * np.cos(N * phi)
    grad = np.array([R - Rc, (-(R - Rc) * dRc - (Z - Zc) * dZc) / R, Z - Zc])
    return np.arcsin(
        abs(tangent @ grad) / np.linalg.norm(tangent) / np.linalg.norm(grad)
    )


def angle_errors(g, result, refs):
    """Errors of the angle between field line and wall of all references"""
    errors = []
    for (prefix, index), (phi, _) in refs.items():
        ix, j, iz = index
        pol, y0 = g.getPoloidalGrid(j)
        angle = exact_angle(pol.R[ix, iz], pol.Z[ix, iz], y0, phi)
        errors.append(abs(result[f"{prefix}_hit_angle"][index] - angle))
    return np.array(errors)


def test_irregular_wall_sections(mms_grid):
    """A wall built from sections with different numbers of points,
    start points and orientations is as accurate as a regular wall"""
    refs = references(mms_grid, hit_on_wall)
    nsections = 64
    phis = np.linspace(0, yperiod, nsections, endpoint=False)
    rng = np.random.default_rng(5)
    lines = []
    for j, phi in enumerate(phis):
        n = int(rng.integers(100, 160))
        theta = np.roll(np.linspace(0, 2 * np.pi, n, endpoint=False), rng.integers(n))
        if j % 2:
            theta = theta[::-1]
        Rc, Zc = wall_centre(phi)
        lines.append(
            rzline.RZline(
                Rc + rw * np.cos(theta), Zc + rw * np.sin(theta), anticlockwise=False
            )
        )
    irregular = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    regular = make_wall(nsections, 128)

    rms, rms_angle = [], []
    for wall in (regular, irregular):
        result = intersection.find_wall_intersections(
            mms_grid, SpiralField(), wall, dphi_max=yperiod / 128
        )
        status, err = errors(result, refs)
        assert np.all(status == intersection.HIT)
        rms.append(np.sqrt(np.mean(err**2)))
        rms_angle.append(np.sqrt(np.mean(angle_errors(mms_grid, result, refs) ** 2)))

        # The crossings lie on the faceted wall
        for prefix in ("forward", "backward"):
            hit = result[f"{prefix}_status"] == intersection.HIT
            for phi, R, Z in zip(
                result[f"{prefix}_hit_phi"][hit],
                result[f"{prefix}_hit_R"][hit],
                result[f"{prefix}_hit_Z"][hit],
            ):
                wall_R, wall_Z = wall.section(phi)
                ring = shapely.LinearRing(np.column_stack((wall_R, wall_Z)))
                assert shapely.distance(ring, shapely.Point(R, Z)) < 1e-9
    # The sections have about the same resolution, so the irregular wall
    # is about as accurate as the regular one
    assert rms[1] < 1.3 * rms[0], rms
    assert rms_angle[1] < 1.3 * rms_angle[0], rms_angle


def coarse_section_wall():
    """Wall of fine sections and one coarse one, with large triangles"""
    nsections = 32
    phis = np.linspace(0, yperiod, nsections, endpoint=False)
    lines = []
    for j, phi in enumerate(phis):
        n = 6 if j == 10 else 64
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        Rc, Zc = wall_centre(phi)
        lines.append(rzline.RZline(Rc + rw * np.cos(theta), Zc + rw * np.sin(theta)))
    return wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)


def test_coarse_section_same_as_brute_force(monkeypatch):
    """The search for triangles near a segment finds all crossings, also
    with triangles of very different size"""
    wall = coarse_section_wall()
    g = make_grid(ny=2, nx=4, nz=8)
    result = intersection.find_wall_intersections(g, SpiralField(), wall)

    def all_pairs(cache, points, radius, lines):
        ntri = len(cache["v0"])
        return np.repeat(lines, ntri), np.tile(np.arange(ntri), len(lines))

    monkeypatch.setattr(intersection, "_candidate_pairs", all_pairs)
    reference = intersection.find_wall_intersections(g, SpiralField(), wall)
    assert np.any(result["forward_status"] == intersection.HIT)
    for key, value in reference.items():
        assert np.array_equal(result[key], value, equal_nan=True), key


def test_large_triangles_do_not_enlarge_search():
    """The few large triangles of a coarse section are searched with their
    own radius, the others with a small one"""
    wall = coarse_section_wall()
    cache = intersection._wall_tree(wall)
    points = cache["centroid"][::7]
    lines = np.arange(len(points))
    _, found = intersection._candidate_pairs(cache, points, 0.0, lines)
    single = cKDTree(cache["centroid"]).query_ball_point(
        points, np.max(cache["radius"])
    )
    assert len(found) < 0.2 * sum(len(f) for f in single)
    # but still all triangles that may touch the points
    distance = (
        np.linalg.norm(cache["centroid"][None] - points[:, None], axis=-1)
        - cache["radius"][None]
    )
    near = np.argwhere(distance <= 0)
    pairs = set(zip(*intersection._candidate_pairs(cache, points, 0.0, lines)))
    assert all((i, t) in pairs for i, t in near)


def test_open_wall(capsys):
    """A wall that covers only part of the period. Grid slices outside it
    are outside the wall, and field lines leaving it don't hit it"""
    nsections = 16
    phis = np.linspace(0.2, 0.6, nsections) * yperiod
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    lines = []
    for phi in phis:
        Rc, Zc = wall_centre(phi)
        lines.append(rzline.RZline(Rc + rw * np.cos(theta), Zc + rw * np.sin(theta)))
    wall = wallgeometry.Wall(lines, phi=phis)
    g = make_grid(ny=4)
    result = intersection.find_wall_intersections(
        g, SpiralField(), wall, dphi_max=0.01, quiet=False
    )
    assert "slice 4 of 4" in capsys.readouterr().out

    for j in range(4):
        pol, y0 = g.getPoloidalGrid(j)
        location = result["start_location"][:, j, :]
        if not phis[0] <= y0 <= phis[-1]:
            assert np.all(location == intersection.OUTSIDE)
            continue
        # Inside the exact wall, away from it by more than the polygons
        # deviate from it
        Rc, Zc = wall_centre(y0)
        distance = np.hypot(pol.R - Rc, pol.Z - Zc) - rw
        certain = np.abs(distance) > 0.005
        assert np.any(certain & (distance < 0)) and np.any(certain & (distance > 0))
        expected = np.where(distance < 0, intersection.INSIDE, intersection.OUTSIDE)
        assert np.array_equal(location[certain], expected[certain])
    for prefix in ("forward", "backward"):
        hit = result[f"{prefix}_status"] == intersection.HIT
        assert np.any(hit)
        assert np.all(result[f"{prefix}_hit_phi"][hit] >= phis[0] - 1e-12)
        assert np.all(result[f"{prefix}_hit_phi"][hit] <= phis[-1] + 1e-12)


def test_no_lines_and_empty_legs():
    wall = make_wall(8, 16)
    empty = np.zeros(0)
    result = intersection.wall_intersections_between(
        SpiralField(), wall, empty, empty, 0.0, 0.1
    )
    assert all(value.size == 0 for value in result.values())
    R = np.array([Ra + 0.1])
    result = intersection.wall_intersections_between(
        SpiralField(), wall, R, np.array([Za]), 0.1, 0.1
    )
    assert result["status"][0] == intersection.NO_HIT
    # No hit: the line stays well inside the wall
    result = intersection.wall_intersections_between(
        SpiralField(), wall, R, np.array([Za]), 0.1, 0.2
    )
    assert result["status"][0] == intersection.NO_HIT
    assert np.isnan(result["hit_length"][0])


def test_hit_direction_and_start_location(mms_grid):
    wall = make_wall(32, 64)
    result = intersection.find_wall_intersections(
        mms_grid, SpiralField(), wall, dphi_max=0.01
    )

    # Field lines drift outwards in the forward direction, so they leave the wall
    forward = result["forward_status"] == intersection.HIT
    assert np.any(forward)
    assert np.all(result["forward_hit_direction"][forward] == 1)
    assert np.all(result["start_location"][forward] == intersection.INSIDE)

    # and enter it in the backward direction
    backward = result["backward_status"] == intersection.HIT
    assert np.any(backward)
    assert np.all(result["backward_hit_direction"][backward] == -1)
    assert np.all(result["start_location"][backward] == intersection.OUTSIDE)

    for prefix in ("forward", "backward"):
        hit = result[f"{prefix}_status"] == intersection.HIT
        assert np.all(
            (result[f"{prefix}_hit_fraction"][hit] > 0)
            & (result[f"{prefix}_hit_fraction"][hit] < 1)
        )
        assert np.all(np.isnan(result[f"{prefix}_hit_length"][~hit]))
        assert np.all(result[f"{prefix}_hit_triangle"][~hit] == -1)


@pytest.mark.parametrize("dphi_max", [0.01, 0.2])
def test_start_on_wall(dphi_max):
    """Points on the wall must not hit it at their starting point.
    Regression: with long segments the refinement fails on the triangle
    of the straight segment more often, and the search of the
    neighbouring triangles found the start point"""
    nsections = 16
    wall = make_wall(nsections, 32)
    # Grid slices on wall sections, grid points on wall vertices
    ycoords = np.linspace(0, yperiod, nsections, endpoint=False)[::4]
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    polys = []
    for y in ycoords:
        Rc, Zc = wall_centre(y)
        scale = np.array([0.7, 0.8, 0.9, 1.0])[:, None]
        polys.append(
            poloidal_grid.StructuredPoloidalGrid(
                Rc + scale * rw * np.cos(theta), Zc + scale * rw * np.sin(theta)
            )
        )
    g = grid.Grid(polys, ycoords, yperiod, yperiodic=True)
    result = intersection.find_wall_intersections(
        g, SpiralField(), wall, dphi_max=dphi_max
    )

    assert np.all(result["start_location"][-1] == intersection.ON_WALL)
    assert np.all(result["start_location"][:-1] == intersection.INSIDE)
    for prefix in ("forward", "backward"):
        hit = result[f"{prefix}_status"] > 0
        assert np.any(hit[-1])
        assert np.all(result[f"{prefix}_hit_length"][hit] > 1e-6)


@pytest.mark.parametrize(
    "refine, status",
    [
        (True, intersection.HIT),
        (False, intersection.HIT),
        ("fails", intersection.HIT_UNREFINED),
    ],
)
def test_crossings_lie_on_the_wall(mms_grid, monkeypatch, refine, status):
    """Also crossings of the straight segment, if not refined or if the
    refinement fails, are points of the faceted wall: regression, their
    toroidal angle was that of the field line, not of the straight segment"""
    if refine == "fails":
        monkeypatch.setattr(
            intersection,
            "_refine_on_plane",
            lambda segments, sel, cache, tri: (
                np.zeros(len(tri)),
                np.zeros(len(tri), bool),
            ),
        )
    wall = make_wall(32, 64)
    result = intersection.find_wall_intersections(
        mms_grid, SpiralField(), wall, dphi_max=0.04, refine=bool(refine)
    )
    for prefix in ("forward", "backward"):
        hit = result[f"{prefix}_status"] != intersection.NO_HIT
        assert np.any(hit)
        assert np.all(result[f"{prefix}_status"][hit] == status)
        for phi, R, Z in zip(
            result[f"{prefix}_hit_phi"][hit],
            result[f"{prefix}_hit_R"][hit],
            result[f"{prefix}_hit_Z"][hit],
        ):
            ring = shapely.LinearRing(np.column_stack(wall.section(phi)))
            assert shapely.distance(ring, shapely.Point(R, Z)) < 1e-9


# Unit triangle in the plane z = 0
TRIANGLE = (np.zeros(3), np.array([1.0, 0, 0]), np.array([0.0, 1, 0]))


@pytest.mark.parametrize(
    "origin, direction, t_expected",
    [
        ([0.2, 0.2, 1], [0, 0, -2], 0.5),
        # On an edge and at a vertex the crossing counts for each triangle
        ([0.5, 0.5, 1], [0, 0, -2], 0.5),
        ([0.5, 0.0, 1], [0, 0, -2], 0.5),
        ([0.0, 0.0, 1], [0, 0, -2], 0.5),
        ([1.0, 0.0, 1], [0, 0, -2], 0.5),
        # The ends of the segment are included
        ([0.2, 0.2, 1], [0, 0, -1], 1.0),
        ([0.2, 0.2, 0], [0, 0, -1], 0.0),
        ([0.2, 0.2, 1], [0.1, -0.1, -2], 0.5),
        # Misses
        ([0.2, 0.2, 1], [0, 0, -0.999], None),
        ([0.2, 0.2, -0.001], [0, 0, -1], None),
        ([0.6, 0.6, 1], [0, 0, -2], None),
        ([-1e-9, 0.2, 1], [0, 0, -2], None),
        # Parallel to the triangle, also in its plane, and zero length
        ([0.2, 0.2, 1], [1, 0, 0], None),
        ([-1.0, 0.2, 0], [2, 0, 0], None),
        ([0.2, 0.2, 0], [0, 0, 0], None),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e3])
def test_segment_triangle(origin, direction, t_expected, scale):
    """Crossings of a segment with a triangle at their boundaries, at
    any length scale"""
    v0, e1, e2 = (scale * x[None] for x in TRIANGLE)
    t, hit = intersection._segment_triangle(
        scale * np.array([origin], dtype=float),
        scale * np.array([direction], dtype=float),
        v0,
        e1,
        e2,
    )
    if t_expected is None:
        assert not hit[0]
    else:
        assert hit[0]
        assert t[0] == pytest.approx(t_expected, abs=1e-12)


def test_crossings_at_vertices_and_edges_are_found():
    """The mesh has no gaps: straight lines through every vertex and edge
    of the sections the wall was built from cross it, also at the period
    boundary. The line ends on the wall in one segment and starts on it in
    the next, also across the period boundary"""
    wall = make_wall(16, 32)
    for k in (0, 5, 16):
        phi = wall.mesh_phi[k]
        R, Z = wall.contours[k % 16].T
        centre = np.mean(R), np.mean(Z)
        for r, z in ((R, Z), (0.5 * (R + np.roll(R, -1)), 0.5 * (Z + np.roll(Z, -1)))):
            # Within the section
            Rs = np.vstack((np.full_like(r, centre[0]), 2 * r - centre[0]))
            Zs = np.vstack((np.full_like(z, centre[1]), 2 * z - centre[1]))
            segment, *_ = intersection.first_crossings(wall, Rs, Zs, np.full(2, phi))
            assert np.all(segment == 0)

            # From inside to outside, through a node on the wall
            phis = phi + np.array([-0.01, 0.0, 0.01])
            Rs = np.vstack((0.5 * (r + centre[0]), r, 1.5 * r - 0.5 * centre[0]))
            Zs = np.vstack((0.5 * (z + centre[1]), z, 1.5 * z - 0.5 * centre[1]))
            segment, *_ = intersection.first_crossings(wall, Rs, Zs, phis)
            assert np.all(segment >= 0)


def test_trace_failure():
    """Lines for which the field can't be evaluated are marked as failed"""

    class BrokenField(SpiralField):
        def Byfunc(self, x, z, phi):
            # Fails above the axis
            return np.where(z > 0.2, 0.0, 1.0)

    g = make_grid(ny=2)
    wall = make_wall(16, 32)
    result = intersection.find_wall_intersections(g, BrokenField(), wall, dphi_max=0.05)
    pol, y0 = g.getPoloidalGrid(0)
    _, y1 = g.getPoloidalGrid(1)
    failed = result["forward_status"][:, 0, :] == intersection.TRACE_FAILED

    # Highest point reached along each field line
    phis = np.linspace(y0, y1, 200)
    _, Z = exact_position(pol.R[..., None], pol.Z[..., None], y0, phis)
    Zmax = Z.max(axis=-1)
    assert np.all(failed[Zmax > 0.2])
    assert not np.any(failed[Zmax < 0.19])
    assert np.any(~failed)


def test_cartesian_field_rejected():
    g = make_grid(ny=2)
    with pytest.raises(ValueError, match="cylindrical"):
        intersection.find_wall_intersections(g, field.Slab(), make_wall(8, 16))


def test_sample_phi_includes_period_boundaries():
    wall = make_wall(8, 16)
    phis = intersection.sample_phi(yperiod - 0.1, yperiod + 0.1, 0.03, wall)
    assert np.any(np.isclose(phis, yperiod, rtol=0, atol=1e-14))
    assert np.all(np.diff(phis) > 0)
    assert np.max(np.diff(phis)) <= 0.03 + 1e-12

    phis = intersection.sample_phi(0.1, -0.1, 0.03, wall)
    assert np.any(np.isclose(phis, 0.0, rtol=0, atol=1e-14))
    assert np.all(np.diff(phis) < 0)


def test_wall_periodic_closure():
    """A periodic wall must not be closed through the inside of the period"""
    wall = make_wall(32, 64)
    # Radial segment in the middle of the period, inside the wall
    phi = 0.5 * yperiod
    Rc, Zc = wall_centre(phi)
    R = np.linspace(Rc - 0.3, Rc + 0.3, 30)
    line = np.column_stack(
        wallgeometry.rzphi_to_xyz(R, np.full(30, Zc), np.full(30, phi))
    )
    assert wallgeometry.first_wall_intersection_fast(wall, line)[0] is None

    # From the axis outwards, the line hits the wall once
    R = np.linspace(Ra, Rc + 0.8, 30)
    line = np.column_stack(
        wallgeometry.rzphi_to_xyz(R, np.full(30, Zc), np.full(30, phi))
    )
    point, segment, triangle, distance = wallgeometry.first_wall_intersection_fast(
        wall, line
    )
    assert np.hypot(point[0], point[1]) == pytest.approx(Rc + rw, abs=0.01)
    assert line[segment, 0] < point[0] < line[segment + 1, 0]
    assert distance == pytest.approx(np.hypot(point[0], point[1]) - Ra)
    assert wall.normals[triangle] @ (line[-1] - line[0]) > 0

    # The mesh covers exactly one period
    assert wall.mesh_phi[0] == 0.0
    assert wall.mesh_phi[-1] == pytest.approx(yperiod)
    assert wall.to_base_period(yperiod + 0.1)[0] == pytest.approx(0.1)
    assert wall.to_base_period(-0.1)[0] == pytest.approx(yperiod - 0.1)


@pytest.mark.parametrize("scale", [1e-2, 1e-3])
def test_first_wall_intersection_independent_of_scale(scale):
    """Regression: the test for segments parallel to a triangle used an
    absolute tolerance, so that all crossings of a small wall were missed"""
    phis = np.linspace(0, yperiod, 32, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    lines = [
        rzline.RZline(scale * (2 + 0.5 * np.cos(theta)), scale * 0.5 * np.sin(theta))
        for _ in phis
    ]
    wall = wallgeometry.Wall(lines, phi=phis, yperiod=yperiod)
    R = scale * np.linspace(2.0, 2.8, 400)
    line = np.column_stack(
        wallgeometry.rzphi_to_xyz(R, np.zeros_like(R), np.full_like(R, 0.5 * yperiod))
    )
    point, *_ = wallgeometry.first_wall_intersection_fast(wall, line)
    assert np.hypot(point[0], point[1]) == pytest.approx(2.5 * scale, rel=1e-9)


def test_start_locations():
    wall = make_wall(8, 16)
    for shift in (0, yperiod, -2 * yperiod):
        phi = wall.mesh_phi[0] + shift
        Rc, Zc = wall_centre(phi)
        location = intersection.start_locations(
            wall,
            np.array([Ra, Ra + 1.0, Rc + rw, Rc + rw - 1e-8, Rc + rw + 1e-8]),
            np.array([Za, Za, Zc, Zc, Zc]),
            phi,
        )
        assert list(location) == [
            intersection.INSIDE,
            intersection.OUTSIDE,
            intersection.ON_WALL,
            intersection.INSIDE,
            intersection.OUTSIDE,
        ]
