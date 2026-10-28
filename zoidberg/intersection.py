"""Intersections of magnetic field lines with a wall

For every point of a grid, the field line is followed to the next
(forward) and previous (backward) poloidal slice. Where the field line
crosses the wall before it reaches that slice, the crossing point, the
parallel length to it and the angle of incidence are calculated.

The wall is a :py:obj:`~zoidberg.wallgeometry.Wall`, which is
independent of the grid and usually has a much finer toroidal
resolution than the grid. The field must be in cylindrical
coordinates, i.e. x = R, y = phi, z = Z.

The field line is sampled with toroidal spacing ``dphi_max`` and the
first segment crossing the wall is found. The crossing is then refined
on a cubic Hermite interpolation of the field line, which uses the field
direction at both ends of the segment. Both the position and the
parallel length therefore converge with fourth order in ``dphi_max``.
Without refinement (``refine=False``) the crossing of the straight
segment is used, which converges with second order.

Example
-------

>>> wall = Wall(wall_lines, phi=wall_phi, yperiod=yperiod)
>>> result = find_wall_intersections(grid, field, wall)
>>> result["forward_hit_length"]

"""

import warnings

import numpy as np
from scipy.integrate import ODEintWarning, odeint
from scipy.spatial import cKDTree

from .wallgeometry import _wall_cache

# Location of a grid point relative to the wall
INSIDE = 0
ON_WALL = 1
OUTSIDE = 2

# Status of a field line
TRACE_FAILED = -1
NO_HIT = 0
HIT = 1
HIT_UNREFINED = 2  # Refinement failed, the crossing of the straight segment is used

# Number of field lines integrated together
_CHUNK_SIZE = 1000


def trace_with_length(magnetic_field, R, Z, phi, rtol=1e-10, atol=1e-12):
    """Follow field lines and integrate the parallel length along them

    Parameters
    ----------
    magnetic_field : :py:obj:`~zoidberg.field.MagneticField`
        Field in cylindrical coordinates
    R, Z : ndarray
        1D arrays of starting positions at phi[0]
    phi : ndarray
        Toroidal angles to output the field line at
    rtol, atol : float
        Tolerances of the integrator

    Returns
    -------
    R, Z, length : ndarray, shape (len(phi), len(R))
        Position of the field lines and parallel length from the start
    failed : ndarray of bool, shape (len(R),)
        Field lines for which the integration failed
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n = R.size

    out = np.zeros((3, len(phi), n))
    failed = np.zeros(n, dtype=bool)
    if n == 0:
        return out[0], out[1], out[2], failed

    # The length increases in the direction of tracing
    direction = np.sign(phi[-1] - phi[0])

    # The state is (R, Z, length) for each line, so that the Jacobian is
    # banded and cheap to estimate if the integrator switches to stiff mode
    def rhs(state, y):
        state = state.reshape(-1, 3)
        dpos = magnetic_field.field_direction(state[:, :2].ravel(), y, flatten=True)
        dpos = dpos.reshape(-1, 2)
        Rmaj = magnetic_field.Rfunc(state[:, 0], state[:, 1], y)
        dlength = np.sqrt(Rmaj**2 + dpos[:, 0] ** 2 + dpos[:, 1] ** 2)
        return np.column_stack((dpos, direction * dlength)).ravel()

    def integrate(sel):
        """Integrate the lines sel, returns True if this failed"""
        state = np.column_stack((R[sel], Z[sel], np.zeros(sel.size))).ravel()
        try:
            with warnings.catch_warnings():
                # Failures are detected from the returned message
                warnings.simplefilter("ignore", ODEintWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                result, info = odeint(
                    rhs, state, phi, rtol=rtol, atol=atol, ml=2, mu=2, full_output=True
                )
            success = "successful" in info["message"]
        except (ValueError, ArithmeticError):
            result = np.full((len(phi), state.size), np.nan)
            success = False
        result = result.reshape(len(phi), -1, 3)
        for i in range(3):
            out[i][:, sel] = result[:, :, i]
        return not (success and np.all(np.isfinite(result)))

    if phi[0] == phi[-1]:
        out[0][:] = R
        out[1][:] = Z
        return out[0], out[1], out[2], failed

    def integrate_or_split(sel):
        if not integrate(sel):
            return
        if sel.size == 1:
            failed[sel] = True
            return
        # Something went wrong, so find out which lines failed
        half = sel.size // 2
        integrate_or_split(sel[:half])
        integrate_or_split(sel[half:])

    for chunk in np.array_split(np.arange(n), max(1, n // _CHUNK_SIZE)):
        integrate_or_split(chunk)
    out[:, :, failed] = np.nan

    return out[0], out[1], out[2], failed


def sample_phi(y0, y1, dphi_max, wall=None):
    """Toroidal angles at which the field line is sampled from y0 to y1

    The spacing is at most dphi_max. If the wall is periodic, the
    period boundaries of the wall are added, so that no segment crosses
    a period boundary.
    """
    nsteps = max(1, int(np.ceil(abs(y1 - y0) / dphi_max - 1e-9)))
    phis = np.linspace(y0, y1, nsteps + 1)
    if wall is None or wall.yperiod is None or y0 == y1:
        return phis

    lo, hi = min(y0, y1), max(y0, y1)
    period = wall.yperiod
    phi0 = wall.mesh_phi[0]
    kmin = np.ceil((lo - phi0) / period)
    kmax = np.floor((hi - phi0) / period)
    bounds = phi0 + np.arange(kmin, kmax + 1) * period
    tol = 1e-9 * max(abs(y1 - y0), 1.0)
    bounds = [b for b in bounds if np.min(np.abs(phis - b)) > tol]
    if not bounds:
        return phis
    phis = np.sort(np.concatenate((phis, bounds)))
    return phis if y1 > y0 else phis[::-1]


def _xyz(R, Z, phi):
    return np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1)


def _segment_triangle(origin, direction, v0, e1, e2):
    """Moller-Trumbore intersection of segments with triangles, pairwise

    Returns the segment parameter t (hit for 0 <= t <= 1) and a mask
    of valid intersections
    """
    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)

    scale = (
        np.linalg.norm(direction, axis=1)
        * np.linalg.norm(e1, axis=1)
        * np.linalg.norm(e2, axis=1)
    )
    valid = np.abs(a) > 1e-14 * scale
    f = np.divide(1.0, a, out=np.zeros_like(a), where=valid)

    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * np.einsum("ij,ij->i", direction, q)
    t = f * np.einsum("ij,ij->i", e2, q)

    eps = 1e-12
    hit = valid & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= 0) & (t <= 1)
    return t, hit


def _wall_tree(wall):
    """Search trees of the triangle centroids. The triangles are grouped
    by size, so that a few large triangles do not enlarge the search
    radius for all others."""
    cache = _wall_cache(wall)
    if "trees" not in cache:
        radius = cache["radius"]
        group = np.floor(np.log2(radius / np.min(radius))).astype(int)
        cache["trees"] = []
        for g in np.unique(group):
            index = np.flatnonzero(group == g)
            tree = cKDTree(cache["centroid"][index])
            cache["trees"].append((tree, index, np.max(radius[index])))
    return cache


def _candidate_pairs(cache, points, radius, lines):
    """Pairs of (line, triangle) with triangles within radius of points"""
    pair_line, pair_tri = [], []
    for tree, index, radius_max in cache["trees"]:
        found = tree.query_ball_point(
            points, radius + radius_max, return_sorted=False, workers=-1
        )
        counts = np.fromiter((len(f) for f in found), dtype=int, count=len(found))
        if counts.sum() > 0:
            pair_line.append(np.repeat(lines, counts))
            pair_tri.append(index[np.concatenate(found).astype(int)])
    if not pair_line:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(pair_line), np.concatenate(pair_tri)


def first_crossings(wall, R, Z, phis, failed=None, min_distance=1e-9):
    """Find the first segment of each sampled field line that crosses the wall

    Parameters
    ----------
    wall : :py:obj:`~zoidberg.wallgeometry.Wall`
    R, Z : ndarray, shape (len(phis), nlines)
        Sampled field lines
    phis : ndarray
        Toroidal angles of the samples
    failed : ndarray of bool, optional
        Lines to skip
    min_distance : float
        Crossings closer than this to the start of the line are ignored,
        so that lines starting on the wall do not hit it immediately [m]

    Returns
    -------
    segment, triangle : ndarray of int
        Index of the segment and the triangle hit, -1 if no hit
    t : ndarray
        Position of the crossing along the straight segment
    shift : ndarray
        Toroidal shift of the segment into the period of the wall mesh
    """
    cache = _wall_tree(wall)
    nlines = R.shape[1]
    segment = np.full(nlines, -1)
    triangle = np.full(nlines, -1)
    t_hit = np.full(nlines, np.nan)
    shift_hit = np.zeros(nlines)

    alive = np.ones(nlines, dtype=bool) if failed is None else ~failed

    for k in range(len(phis) - 1):
        lines = np.flatnonzero(alive)
        if lines.size == 0:
            break
        # Rotate the segment into the period of the wall
        _, shift = wall.to_base_period(0.5 * (phis[k] + phis[k + 1]))
        p0 = _xyz(R[k, lines], Z[k, lines], phis[k] - shift)
        p1 = _xyz(R[k + 1, lines], Z[k + 1, lines], phis[k + 1] - shift)
        direction = p1 - p0
        seglen = np.linalg.norm(direction, axis=1)

        pair_line, pair_tri = _candidate_pairs(
            cache, 0.5 * (p0 + p1), 0.5 * seglen, np.arange(lines.size)
        )
        if pair_line.size == 0:
            continue

        t, hit = _segment_triangle(
            p0[pair_line],
            direction[pair_line],
            cache["v0"][pair_tri],
            cache["edge1"][pair_tri],
            cache["edge2"][pair_tri],
        )
        if k == 0:
            hit &= t * seglen[pair_line] >= min_distance
        if not np.any(hit):
            continue

        pair_line, pair_tri, t = pair_line[hit], pair_tri[hit], t[hit]
        # Keep the first crossing of each line
        order = np.lexsort((t, pair_line))
        _, first = np.unique(pair_line[order], return_index=True)
        first = order[first]

        hit_lines = lines[pair_line[first]]
        segment[hit_lines] = k
        triangle[hit_lines] = pair_tri[first]
        t_hit[hit_lines] = t[first]
        shift_hit[hit_lines] = shift
        alive[hit_lines] = False

    return segment, triangle, t_hit, shift_hit


def _hermite(u, h, p0, m0, p1, m1):
    """Cubic Hermite interpolation and its derivative w.r.t. the
    original variable, with u in [0, 1] and interval length h"""
    u2 = u * u
    u3 = u2 * u
    value = (
        (2 * u3 - 3 * u2 + 1) * p0
        + (u3 - 2 * u2 + u) * h * m0
        + (-2 * u3 + 3 * u2) * p1
        + (u3 - u2) * h * m1
    )
    deriv = (
        (6 * u2 - 6 * u) * p0 / h
        + (3 * u2 - 4 * u + 1) * m0
        + (-6 * u2 + 6 * u) * p1 / h
        + (3 * u2 - 2 * u) * m1
    )
    return value, deriv


class _Segments:
    """Hermite representation of one segment of each hit field line"""

    def __init__(self, magnetic_field, R, Z, S, phis, segment, shift):
        k = segment
        lines = np.arange(len(k))
        self.phi0 = phis[k]
        self.h = phis[k + 1] - phis[k]
        self.shift = shift
        self.ends = []
        for kk in (k, k + 1):
            r = R[kk, lines]
            z = Z[kk, lines]
            dr = np.zeros_like(r)
            dz = np.zeros_like(z)
            # Field direction at the ends. Evaluated for each angle separately
            for phi in np.unique(phis[kk]):
                sel = phis[kk] == phi
                pos = np.column_stack((r[sel], z[sel])).ravel()
                d = magnetic_field.field_direction(pos, phi, flatten=True)
                dr[sel] = d[0::2]
                dz[sel] = d[1::2]
            # The length increases in the direction of tracing
            ds = np.sqrt(magnetic_field.Rfunc(r, z, phis[kk]) ** 2 + dr**2 + dz**2)
            ds *= np.sign(self.h)
            self.ends.append((r, z, S[kk, lines], dr, dz, ds))

    def evaluate(self, u, sel=slice(None)):
        """R, Z, length, dR/dphi, dZ/dphi at fractions u of the segments"""
        (r0, z0, s0, dr0, dz0, ds0), (r1, z1, s1, dr1, dz1, ds1) = self.ends
        h = self.h[sel]
        r, dr = _hermite(u, h, r0[sel], dr0[sel], r1[sel], dr1[sel])
        z, dz = _hermite(u, h, z0[sel], dz0[sel], z1[sel], dz1[sel])
        s, _ = _hermite(u, h, s0[sel], ds0[sel], s1[sel], ds1[sel])
        return r, z, s, dr, dz

    def xyz(self, u, sel=slice(None)):
        """Position in the rotated frame of the wall mesh"""
        r, z, *_ = self.evaluate(u, sel)
        phi = self.phi0[sel] + u * self.h[sel] - self.shift[sel]
        return _xyz(r, z, phi)


def _inside_triangle(p, v0, e1, e2, tol=1e-9):
    """Is the point p (in the plane of the triangle) inside it?"""
    w = p - v0
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", w, e1)
    d21 = np.einsum("ij,ij->i", w, e2)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w_ = (d00 * d21 - d01 * d20) / denom
    return (v >= -tol) & (w_ >= -tol) & (v + w_ <= 1 + tol)


def _refine_on_plane(segments, sel, cache, tri, niter=60):
    """Find where the Hermite field line crosses the plane of each triangle

    Returns the fraction u along the segment, and whether the crossing
    was bracketed and lies inside the triangle
    """
    normal = np.cross(cache["edge1"][tri], cache["edge2"][tri])
    v0 = cache["v0"][tri]

    def f(u):
        return np.einsum("ij,ij->i", normal, segments.xyz(u, sel) - v0)

    lo = np.zeros(len(tri))
    hi = np.ones(len(tri))
    f_lo = f(lo)
    f_hi = f(hi)
    ok = f_lo * f_hi <= 0
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        left = f_lo * f_mid <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    u = 0.5 * (lo + hi)
    ok &= _inside_triangle(
        segments.xyz(u, sel), v0, cache["edge1"][tri], cache["edge2"][tri]
    )
    return u, ok


def refine_crossings(
    magnetic_field, wall, R, Z, S, phis, segment, triangle, t, shift, min_distance=1e-9
):
    """Refine the crossing points found by `first_crossings`

    Crossings on the first segment closer than min_distance to the
    start of the line are ignored, as in `first_crossings`.

    Returns
    -------
    u : ndarray
        Fraction along the hit segment of the crossing
    triangle : ndarray
        Triangle that is crossed
    refined : ndarray of bool
        False where the refinement failed. There u is the crossing of
        the straight segment
    segments : _Segments
        Interpolation of the hit segments
    """
    cache = _wall_tree(wall)
    segments = _Segments(magnetic_field, R, Z, S, phis, segment, shift)
    nhit = len(segment)
    all_hits = np.arange(nhit)
    start = segments.xyz(np.zeros(nhit))

    def away_from_start(u, sel):
        """A line starting on the wall lies in the planes of the triangles
        there, which the bisection would return as crossings"""
        distance = np.linalg.norm(segments.xyz(u, sel) - start[sel], axis=1)
        return (segment[sel] > 0) | (distance >= min_distance)

    u, refined = _refine_on_plane(segments, all_hits, cache, triangle)
    refined &= away_from_start(u, all_hits)
    triangle = triangle.copy()

    # The curved field line can cross a neighbouring triangle instead.
    # Try all triangles close to the first guess
    retry = np.flatnonzero(~refined)
    if retry.size > 0:
        guess = segments.xyz(t[retry], retry)
        seglen = np.linalg.norm(
            segments.xyz(np.ones(retry.size), retry)
            - segments.xyz(np.zeros(retry.size), retry),
            axis=1,
        )
        pair_line, pair_tri = _candidate_pairs(cache, guess, seglen, retry)
        if pair_line.size > 0:
            u_pair, ok = _refine_on_plane(segments, pair_line, cache, pair_tri)
            ok &= away_from_start(u_pair, pair_line)
            pair_line, pair_tri, u_pair = pair_line[ok], pair_tri[ok], u_pair[ok]
            order = np.lexsort((u_pair, pair_line))
            _, first = np.unique(pair_line[order], return_index=True)
            first = order[first]
            lines = pair_line[first]
            u[lines] = u_pair[first]
            triangle[lines] = pair_tri[first]
            refined[lines] = True

    u = np.where(refined, u, t)
    return u, triangle, refined, segments


def start_locations(wall, R, Z, phi, tol=1e-9):
    """Location of points at toroidal angle phi relative to the wall

    Returns an array of INSIDE, ON_WALL or OUTSIDE
    """
    import shapely

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    try:
        wall_R, wall_Z = wall.section(phi)
    except ValueError:
        return np.full(R.shape, OUTSIDE)
    polygon = shapely.Polygon(np.column_stack((wall_R, wall_Z)))
    points = shapely.points(R, Z)
    location = np.where(shapely.contains_xy(polygon, R, Z), INSIDE, OUTSIDE)
    location[shapely.distance(polygon.exterior, points) < tol] = ON_WALL
    return location


def wall_intersections_between(
    magnetic_field,
    wall,
    R,
    Z,
    y0,
    y1,
    dphi_max=0.01,
    rtol=1e-10,
    atol=1e-12,
    refine=True,
    min_distance=1e-9,
):
    """Intersections with the wall of field lines from y0 to y1

    Parameters
    ----------
    magnetic_field : :py:obj:`~zoidberg.field.MagneticField`
        Field in cylindrical coordinates
    wall : :py:obj:`~zoidberg.wallgeometry.Wall`
    R, Z : ndarray
        1D arrays of starting points at toroidal angle y0
    y0, y1 : float
        Start and end toroidal angle
    dphi_max : float
        Maximum toroidal step between samples of the field line
    rtol, atol : float
        Tolerances of the field line integration
    refine : bool
        Refine the crossing on a cubic interpolation of the field line.
        Otherwise the crossing of the straight segment is used.
    min_distance : float
        Crossings closer than this to the start point are ignored [m]

    Returns
    -------
    dict
        status, hit_fraction, hit_phi, hit_R, hit_Z, hit_length,
        hit_angle, hit_direction and hit_triangle, as 1D arrays.
        See `find_wall_intersections`
    """
    if magnetic_field.Rfunc(np.ones(1), np.zeros(1), 0.0) is None:
        raise ValueError("Wall intersections need a field in cylindrical coordinates")

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = R.size

    result = {
        "status": np.full(n, NO_HIT),
        "hit_fraction": np.full(n, np.nan),
        "hit_phi": np.full(n, np.nan),
        "hit_R": np.full(n, np.nan),
        "hit_Z": np.full(n, np.nan),
        "hit_length": np.full(n, np.nan),
        "hit_angle": np.full(n, np.nan),
        "hit_direction": np.zeros(n, dtype=int),
        "hit_triangle": np.full(n, -1),
    }
    if y0 == y1 or n == 0:
        return result

    phis = sample_phi(y0, y1, dphi_max, wall)
    Rs, Zs, Ss, failed = trace_with_length(
        magnetic_field, R, Z, phis, rtol=rtol, atol=atol
    )
    result["status"][failed] = TRACE_FAILED

    segment, triangle, t, shift = first_crossings(
        wall, Rs, Zs, phis, failed=failed, min_distance=min_distance
    )
    hit = np.flatnonzero(segment >= 0)
    if hit.size == 0:
        return result

    # Crossing of the straight segment
    k = segment[hit]
    t = t[hit]
    p0 = _xyz(Rs[k, hit], Zs[k, hit], phis[k])
    p1 = _xyz(Rs[k + 1, hit], Zs[k + 1, hit], phis[k + 1])
    chord = p0 + t[:, None] * (p1 - p0)
    chord_r = np.hypot(chord[:, 0], chord[:, 1])
    chord_s = Ss[k, hit] + t * (Ss[k + 1, hit] - Ss[k, hit])
    # The chord is not at phis[k] + t * h, so turn from p0 to the crossing
    chord_phi = phis[k] + np.arctan2(
        p0[:, 0] * chord[:, 1] - p0[:, 1] * chord[:, 0],
        p0[:, 0] * chord[:, 0] + p0[:, 1] * chord[:, 1],
    )

    if refine:
        u, triangle_hit, refined, segments = refine_crossings(
            magnetic_field,
            wall,
            Rs[:, hit],
            Zs[:, hit],
            Ss[:, hit],
            phis,
            k,
            triangle[hit],
            t,
            shift[hit],
            min_distance=min_distance,
        )
        r, z, s, dr, dz = segments.evaluate(u)
        # Where the refinement failed, use the straight segment
        r = np.where(refined, r, chord_r)
        z = np.where(refined, z, chord[:, 2])
        s = np.where(refined, s, chord_s)
        phi_hit = np.where(refined, segments.phi0 + u * segments.h, chord_phi)
        result["status"][hit] = np.where(refined, HIT, HIT_UNREFINED)
    else:
        u = t
        triangle_hit = triangle[hit]
        segments = _Segments(
            magnetic_field, Rs[:, hit], Zs[:, hit], Ss[:, hit], phis, k, shift[hit]
        )
        _, _, _, dr, dz = segments.evaluate(u)
        r, z, s = chord_r, chord[:, 2], chord_s
        phi_hit = chord_phi
        result["status"][hit] = HIT

    # Angle between the field line and the wall surface
    phi_b = phi_hit - segments.shift
    tangent = np.stack(
        (
            dr * np.cos(phi_b) - r * np.sin(phi_b),
            dr * np.sin(phi_b) + r * np.cos(phi_b),
            dz,
        ),
        axis=-1,
    ) * np.sign(y1 - y0)
    normal = wall.normals[triangle_hit]
    tn = np.einsum("ij,ij->i", tangent, normal)

    result["hit_fraction"][hit] = (phi_hit - y0) / (y1 - y0)
    result["hit_phi"][hit] = phi_hit
    result["hit_R"][hit] = r
    result["hit_Z"][hit] = z
    result["hit_length"][hit] = s
    result["hit_angle"][hit] = np.arcsin(
        np.clip(np.abs(tn) / np.linalg.norm(tangent, axis=1), 0, 1)
    )
    result["hit_direction"][hit] = np.sign(tn).astype(int)
    result["hit_triangle"][hit] = triangle_hit
    return result


def find_wall_intersections(grid, magnetic_field, wall, quiet=True, **kwargs):
    """Intersections of the field lines from every grid point with the wall

    Each field line is followed from its poloidal slice to the next
    slice (forward) and to the previous slice (backward). All points
    are traced, including those outside the wall.

    Parameters
    ----------
    grid : :py:obj:`~zoidberg.grid.Grid`
        Any grid; it does not need to be related to the wall
    magnetic_field : :py:obj:`~zoidberg.field.MagneticField`
        Field in cylindrical coordinates (x = R, y = phi, z = Z)
    wall : :py:obj:`~zoidberg.wallgeometry.Wall`
        Wall with a mesh, see `Wall.create_mesh`
    quiet : bool
        Don't print progress
    kwargs
        Passed to `wall_intersections_between`, e.g. dphi_max, rtol,
        atol, refine, min_distance. ``on_wall_tol`` [m] sets the distance
        within which a grid point counts as lying on the wall.

    Returns
    -------
    dict
        Arrays of shape (nx, ny, nz):

        - **start_location**: INSIDE (0), ON_WALL (1) or OUTSIDE (2) of
          the grid point relative to the wall

        and for prefix "forward" and "backward":

        - **<prefix>_status**: NO_HIT (0), HIT (1), HIT_UNREFINED (2)
          or TRACE_FAILED (-1)
        - **<prefix>_hit_fraction**: (phi_hit - phi) / (phi_next - phi)
        - **<prefix>_hit_phi, _hit_R, _hit_Z**: position of the crossing
        - **<prefix>_hit_length**: parallel length from the grid point to
          the crossing [m]
        - **<prefix>_hit_angle**: angle between the field line and the
          wall surface [radians]; 0 is grazing
        - **<prefix>_hit_direction**: +1 if the field line leaves the
          wall there, -1 if it enters
        - **<prefix>_hit_triangle**: index of the wall triangle

        Values without a crossing are NaN, or -1 / 0 for integers.
    """
    on_wall_tol = kwargs.pop("on_wall_tol", 1e-9)

    nx, ny, nz = grid.shape
    result = {"start_location": np.full((nx, ny, nz), OUTSIDE)}

    for j in range(ny):
        pol, y0 = grid.getPoloidalGrid(j)
        R = pol.R.ravel()
        Z = pol.Z.ravel()
        result["start_location"][:, j, :] = start_locations(
            wall, R, Z, y0, tol=on_wall_tol
        ).reshape(nx, nz)

        for prefix, offset in (("forward", 1), ("backward", -1)):
            _, y1 = grid.getPoloidalGrid(j + offset)
            leg = wall_intersections_between(
                magnetic_field, wall, R, Z, y0, y1, **kwargs
            )
            for key, value in leg.items():
                name = f"{prefix}_{key}"
                if name not in result:
                    result[name] = np.empty((nx, ny, nz), dtype=value.dtype)
                result[name][:, j, :] = value.reshape(nx, nz)

        if not quiet:
            print(f"Wall intersections: slice {j + 1} of {ny}", flush=True)

    return result
