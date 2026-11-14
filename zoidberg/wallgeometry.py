"""Triangulated 3D wall built from poloidal cross-sections of the wall

The wall is given as a set of closed (R, Z) contours at a list of
toroidal angles. Neighbouring contours are joined into a triangle mesh
in Cartesian (X, Y, Z) coordinates, so that all intersection tests are
done in real space.

The wall resolution is independent of the grid: typically the wall is
built from many more cross-sections than the grid has poloidal slices.

Each strip between two neighbouring contours is triangulated on its own
(lofting). The contours may have different numbers of points, different
start points and different point spacing; the points of the contours are
the vertices of the mesh. Which points are joined is found in two steps:

1. A transform that maps one contour onto the other is fitted by
   trimmed iterative closest points: a translation and uniform scaling,
   and a rotation if the contours differ mainly by one, as an elongated
   cross-section turning around. It follows the motion of the contour as
   a whole and ignores local differences, for example a port that
   appears.
2. The edges between the contours (rungs) minimise the summed squared
   distance between the points of one contour and the transformed points
   of the other, by dynamic programming (Fuchs, Kedem & Uselton, Commun.
   ACM 20, 693 (1977)).

As the points are kept, neighbouring cross-sections should have a
similar poloidal resolution: between a coarse and a fine polygon the
facets follow the coarse polygon on one side and the fine one on the
other, so that they are tilted by about the deviation between the two
polygons divided by the toroidal distance of the sections. This affects
the angle between field lines and the wall.

"""

import warnings

import numpy as np

# Padded cells of the dynamic programming tables processed at once
_DP_BATCH_CELLS = 2**28

# Weight of the squared distance in R, Z relative to that after aligning
# the contours, in the cost of the rungs
_REAL_WEIGHT = 0.01

# Fractions of each strip at which Wall.check checks the cross-section
_CHECK_FRACTIONS = tuple((np.arange(16) + 0.5) / 16)


def rzphi_to_xyz(r, z, phi):
    return r * np.cos(phi), r * np.sin(phi), z


def _signed_area(r, z):
    """Signed area of closed (r, z) polygons along the last axis.
    Positive if anticlockwise in the (R, Z) plane"""
    return 0.5 * np.sum(
        r * np.roll(z, -1, axis=-1) - np.roll(r, -1, axis=-1) * z, axis=-1
    )


def clean_contour(R, Z):
    """Check a closed contour and orient it anticlockwise in (R, Z)

    Repeated consecutive points, including a last point equal to the
    first, are removed. Points closer than 1e-9 times the largest
    coordinate count as repeated.

    Parameters
    ----------
    R, Z : array_like
        Points of the contour

    Returns
    -------
    ndarray, shape (n_points, 2)

    Raises
    ------
    ValueError
        If the contour has fewer than three distinct points, is not
        finite, is not at R > 0, has no area or intersects itself
    """
    import shapely

    P = np.column_stack((np.ravel(R), np.ravel(Z))).astype(float)
    if not np.all(np.isfinite(P)):
        raise ValueError("Wall contour contains non-finite points")
    if np.any(P[:, 0] <= 0):
        raise ValueError("Wall contour must lie at R > 0")
    scale = np.max(np.abs(P))
    step = np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1)
    P = P[step > 1e-9 * scale]
    if len(P) < 3:
        raise ValueError("Wall contour needs at least three distinct points")
    area = _signed_area(P[:, 0], P[:, 1])
    if abs(area) <= 1e-12 * scale**2:
        raise ValueError("Wall contour has no area")
    if not shapely.LinearRing(P).is_simple:
        raise ValueError("Wall contour intersects itself")
    return P if area > 0 else P[::-1].copy()


def _arc_fraction(P):
    """Arc length fraction of the points of a closed contour"""
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)[:-1])) / np.sum(seg)


def _resample(P, s, M):
    """Closed contour P with arc length fractions s, at M uniformly
    spaced arc length fractions"""
    u = np.arange(M) / M
    s_closed = np.append(s, 1.0)
    P_closed = np.vstack((P, P[:1]))
    return np.column_stack([np.interp(u, s_closed, P_closed[:, k]) for k in (0, 1)])


def _similarity_fit(P, Q, rotate=True):
    """Transform x -> scale * rotation @ x + shift, without reflection,
    that maps the points Q closest to the points P (least squares).
    If not rotate, the rotation is the identity."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    p, q = P - cp, Q - cq
    if rotate:
        U, S, Vt = np.linalg.svd(q.T @ p)
        d = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        rotation = Vt.T @ d @ U.T
        scale = np.sum(S * np.diag(d)) / np.sum(q * q)
    else:
        rotation = np.eye(2)
        scale = np.sum(p * q) / np.sum(q * q)
    return scale, rotation, cp - scale * rotation @ cq


class _DenseContour:
    """Closed contour sampled densely, with normals, for distances of
    points to it"""

    def __init__(self, P, M):
        from scipy.spatial import cKDTree

        self.nodes = _resample(P, _arc_fraction(P), M)
        tangent = np.roll(self.nodes, -1, axis=0) - np.roll(self.nodes, 1, axis=0)
        self.normals = (
            tangent[:, ::-1] * [1, -1] / np.linalg.norm(tangent, axis=1, keepdims=True)
        )
        self.tree = cKDTree(self.nodes)

    def closest(self, X):
        """Closest nodes to the points X, their normals, and the distances
        of X to the tangent lines there"""
        _, idx = self.tree.query(X)
        nodes, normals = self.nodes[idx], self.normals[idx]
        return nodes, normals, np.abs(np.sum((X - nodes) * normals, axis=1))


def _align(A, B, M=256, keep=0.5, max_iterations=100, rotation_gain=10):
    """Transform that maps contour B onto contour A

    A translation and uniform scaling, with a rotation if the contours
    differ mainly by one: the rotation is used if it reduces the distance
    between the contours by a factor rotation_gain. Otherwise the points
    of B would slide along the contour, which places the triangles inside
    the wall where it is curved.

    Each transform is fitted by iterative closest points, minimising the
    distances of points of A to the tangents of B (point to line). Only
    the closest pairs are used in each iteration: a fraction `keep` of the
    points, of the longer contour only as many as the shorter one has, so
    that local differences between the contours do not affect it. It is
    started from the best overlay of the contours by arc length
    (cross-correlation) and from the identity, and the fit with the
    smaller distance between the contours is used.

    Returns
    -------
    function
        Mapping (n, 2) points of B onto A
    """
    # Sample from a point that does not depend on the numbering
    A = np.roll(A, -np.lexsort((A[:, 1], A[:, 0]))[-1], axis=0)
    B = np.roll(B, -np.lexsort((B[:, 1], B[:, 0]))[-1], axis=0)
    a = _resample(A, _arc_fraction(A), M)
    b = _resample(B, _arc_fraction(B), M)
    dense_a, dense_b = _DenseContour(A, 4 * M), _DenseContour(B, 4 * M)
    size = np.sqrt(abs(_signed_area(A[:, 0], A[:, 1])))
    length_a = np.sum(np.linalg.norm(np.roll(A, -1, axis=0) - A, axis=1))
    length_b = np.sum(np.linalg.norm(np.roll(B, -1, axis=0) - B, axis=1))

    def nkeep(scale):
        """Numbers of samples of A and of B used. Of the longer contour,
        at most the length of the shorter one can match"""
        ratio = scale * length_b / length_a
        return (
            max(3, int(keep * min(1, ratio) * M)),
            max(3, int(keep * min(1, 1 / ratio) * M)),
        )

    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    corr = sum(
        np.fft.ifft(np.conj(np.fft.fft(ac[:, k])) * np.fft.fft(bc[:, k])).real
        for k in (0, 1)
    )
    b_overlay = np.roll(b, -np.argmax(corr), axis=0)

    def pairs(scale, rotation, shift):
        """Closest points y of the transformed B to the points of A,
        normals there, and distances"""
        nodes, normals, _ = dense_b.closest((a - shift) @ rotation / scale)
        y = scale * nodes @ rotation.T + shift
        n = normals @ rotation.T
        return y, n, np.sum((a - y) * n, axis=1)

    def icp(rotate, start):
        """Distance between the contours and transform, from start"""
        scale, rotation, shift = start
        for _ in range(max_iterations):
            y, n, res = pairs(scale, rotation, shift)
            sel = np.abs(res) <= np.sort(np.abs(res))[nkeep(scale)[0] - 1]
            y, n, res = y[sel], n[sel], res[sel]
            # Small change y -> c + (1 + ds) R(dtheta) (y - c) + dt
            c = y.mean(axis=0)
            columns = [np.sum((y - c) * n, axis=1), n[:, 0], n[:, 1]]
            if rotate:
                columns.append(np.sum((y - c)[:, ::-1] * [-1, 1] * n, axis=1))
            step = np.linalg.lstsq(np.column_stack(columns), res, rcond=None)[0]
            ds, dt = step[0], step[1:3]
            dtheta = step[3] if rotate else 0.0
            turn = np.array(
                [
                    [np.cos(dtheta), -np.sin(dtheta)],
                    [np.sin(dtheta), np.cos(dtheta)],
                ]
            )
            scale = scale * (1 + ds)
            rotation = turn @ rotation
            shift = c + (1 + ds) * turn @ (shift - c) + dt
            if abs(ds) + abs(dtheta) + np.max(np.abs(dt)) / size < 1e-12:
                break

        # Distance of the closest parts of each contour to the other
        n_ab, n_ba = nkeep(scale)
        d_ab = np.abs(pairs(scale, rotation, shift)[2])
        d_ba = dense_a.closest(scale * b @ rotation.T + shift)[2]
        distance = np.mean(np.sort(d_ab)[:n_ab] ** 2) + np.mean(
            np.sort(d_ba)[:n_ba] ** 2
        )
        return distance, scale, rotation, shift

    def fit(rotate):
        """The better fit from the two starts"""
        starts = (_similarity_fit(a, b_overlay, rotate), (1.0, np.eye(2), 0.0))
        return min((icp(rotate, start) for start in starts), key=lambda f: f[0])

    without = fit(rotate=False)
    with_rotation = fit(rotate=True)
    use = with_rotation if with_rotation[0] * rotation_gain < without[0] else without
    _, scale, rotation, shift = use
    return lambda P: scale * P @ rotation.T + shift


def _dp_batch(problems):
    """Optimal monotone paths for a batch of strips

    Parameters
    ----------
    problems : list of (pa, pb)
        pa : (m + 1, 2) points of A in the order of the path, first
            point repeated at the end
        pb : (n + 1, 2) points of B, the same

    Returns
    -------
    paths : list of (m + n + 1, 2) int arrays of (i, j)
        From (0, 0) to (m, n), minimising the sum over the nodes,
        except the first, of the squared distance |pa[i] - pb[j]|^2
    costs : ndarray
        Total cost of each path
    """
    nb = len(problems)
    m = np.array([len(pa) - 1 for pa, _ in problems])
    n = np.array([len(pb) - 1 for _, pb in problems])
    mmax, nmax = m.max(), n.max()

    # Padding beyond n does not change columns <= n, as D[i, j] only
    # depends on columns <= j. Rows beyond m are never used.
    ndim = problems[0][0].shape[1]
    PA = np.zeros((ndim, nb, mmax + 1))
    PB = np.zeros((ndim, nb, nmax + 1))
    for k, (pa, pb) in enumerate(problems):
        PA[:, k, : len(pa)] = pa.T
        PB[:, k, : len(pb)] = pb.T

    def row_cost(i):
        cost = np.zeros((nb, nmax + 1))
        for d in range(ndim):
            diff = PA[d, :, i, None] - PB[d]
            cost += diff * diff
        return cost

    # D[i, j] = cost[i, j] + min(D[i - 1, j], D[i, j - 1]), the minimal
    # cost of a path from (0, 0) to (i, j). Bit j of from_a[:, i] is
    # set if the optimal path reaches (i, j) from (i - 1, j)
    from_a = np.zeros((nb, mmax + 1, (nmax + 8) // 8), dtype=np.uint8)
    total = np.empty(nb)
    batch = np.arange(nb)
    row = np.empty((nb, nmax + 1), dtype=bool)

    D = np.cumsum(row_cost(0), axis=1)
    D -= D[:, :1]
    total[m == 0] = D[m == 0, n[m == 0]]
    for i in range(1, mmax + 1):
        cost = row_cost(i)
        C = np.cumsum(cost, axis=1)
        D_new = C + np.minimum.accumulate(D - C + cost, axis=1)
        row[:, 0] = True
        row[:, 1:] = D[:, 1:] <= D_new[:, :-1]
        from_a[:, i] = np.packbits(row, axis=1)
        D = D_new
        done = m == i
        total[done] = D[batch[done], n[done]]

    # Backtrack. Every step reduces i + j by one, so the node with
    # i + j = p is the p-th node of the path
    i, j = m.copy(), n.copy()
    nodes = np.zeros((nb, mmax + nmax + 1, 2), dtype=int)
    nodes[batch, i + j] = np.column_stack((i, j))
    for _ in range(mmax + nmax):
        active = i + j > 0
        bit = (from_a[batch, i, j >> 3] >> (7 - (j & 7))) & 1
        step_a = active & (bit == 1)
        step_b = active & ~step_a
        i = i - step_a
        j = j - step_b
        nodes[batch[active], (i + j)[active]] = np.column_stack((i, j))[active]

    paths = [nodes[k, : m[k] + n[k] + 1] for k in range(nb)]
    return paths, total


def _strip_points(A, B):
    """Points of contours A and B whose squared distances are the cost of
    the rungs: B aligned to A, and the real coordinates weighted by
    _REAL_WEIGHT"""
    weight = np.sqrt(_REAL_WEIGHT)
    return (
        np.column_stack((A, weight * A)),
        np.column_stack((_align(A, B)(B), weight * B)),
    )


def tile_contours(contours, pairs):
    """Triangulate the strips between pairs of contours

    Parameters
    ----------
    contours : list of ndarray, shape (n_points, 2)
        Anticlockwise contours, see `clean_contour`
    pairs : list of (a, b)
        Indices of the two contours of each strip

    Returns
    -------
    list of ndarray, shape (m + n + 1, 2)
        For each strip the rungs (i, j), pairs of points of contour a
        and b joined by an edge, in order around the strip. The first
        and last rung are the same. Each step between rungs advances
        either i or j by one point and forms one triangle.
    """
    # Contour b of each strip, mapped onto contour a
    points = [_strip_points(contours[ia], contours[ib]) for ia, ib in pairs]

    def solve(starts):
        """Optimal rungs of each strip, with the rung (i0, j0) fixed"""
        problems = []
        for k, i0, j0 in starts:
            A, B = points[k]
            pa = A[(i0 + np.arange(len(A) + 1)) % len(A)]
            pb = B[(j0 + np.arange(len(B) + 1)) % len(B)]
            # The Python loop of the dynamic programming runs over the
            # rows, so use the shorter contour for them
            problems.append((pb, pa, True) if len(A) > len(B) else (pa, pb, False))

        # Batches of similar size, limited by the padded table size
        order = sorted(
            range(len(problems)),
            key=lambda p: (len(problems[p][0]), len(problems[p][1])),
        )
        paths = [None] * len(problems)
        first = 0
        while first < len(order):
            last = first + 1
            cols = len(problems[order[first]][1])
            while last < len(order):
                # Rows increase along order, so the last has the most
                rows = len(problems[order[last]][0])
                new_cols = max(cols, len(problems[order[last]][1]))
                if (last + 1 - first) * rows * new_cols > _DP_BATCH_CELLS:
                    break
                cols = new_cols
                last += 1
            batch = order[first:last]
            result, _ = _dp_batch([problems[p][:2] for p in batch])
            for p, path in zip(batch, result):
                paths[p] = path[:, ::-1] if problems[p][2] else path
            first = last

        rungs = []
        for (k, i0, j0), path in zip(starts, paths):
            m, n = len(contours[pairs[k][0]]), len(contours[pairs[k][1]])
            rungs.append(
                np.column_stack(((i0 + path[:, 0]) % m, (j0 + path[:, 1]) % n))
            )
        return rungs

    # Fix the rung of the first point of A to the closest point of B
    first_pass = solve(
        [
            (k, 0, int(np.argmin(np.sum((B - A[0]) ** 2, axis=1))))
            for k, (A, B) in enumerate(points)
        ]
    )

    # The fixed rung biases the path near the first point of A. Solve
    # again with the rung of the opposite point of A fixed, which the
    # optimal path is not sensitive to.
    second = []
    for k, (ia, _) in enumerate(pairs):
        i0 = len(contours[ia]) // 2
        rungs = first_pass[k]
        second.append((k, i0, rungs[np.argmax(rungs[:, 0] == i0), 1]))
    return solve(second)


def _strip_triangles(a, b):
    """Triangles of a strip from its rungs (a, b), first rung repeated at
    the end. A step along A gives (a_i, a_i+1, b_j), along B
    (a_i, b_j+1, b_j)"""
    step_a = (a[1:] != a[:-1])[:, None]
    return np.where(
        step_a,
        np.column_stack((a[:-1], a[1:], b[:-1])),
        np.column_stack((a[:-1], b[1:], b[:-1])),
    )


def create_wall_mesh(contours, phi, close_toroidal=False, yperiod=None):
    """
    Create a triangular surface mesh from R-Z contours.

    Parameters
    ----------
    contours : list of ndarray, shape (n_points, 2)
        Anticlockwise (R, Z) contours, see `clean_contour`. The number
        of points may differ between contours.

    phi : ndarray, shape (n_sections,)
        Toroidal angles in radians.

    close_toroidal : bool
        Connect the last cross-section directly to the first.
        Use True only when the sections cover a full 2*pi torus.

    yperiod : float, optional
        Toroidal period of the wall, e.g. 2*pi/5 for a five-period
        stellarator. If given, the first cross-section is repeated at
        phi[0] + yperiod and connected to the last one, so that the mesh
        covers exactly one period. Overrides close_toroidal.

    The input is not checked; `Wall` checks it and the mesh.

    Returns
    -------
    vertices : ndarray, shape (n_vertices, 3)
        Cartesian coordinates of the vertices, the points of all
        contours in order
    triangles : ndarray, shape (n_triangles, 3)
        Vertex indices of each triangle, strip by strip
    strip_rungs : list of ndarray, shape (m + n, 2)
        Vertex indices of the rungs of each strip in order, see
        `tile_contours`; the closing rung is not repeated
    """
    contours = [np.asarray(c, dtype=float) for c in contours]
    phi = np.asarray(phi, dtype=float)

    if len(phi) != len(contours):
        raise ValueError("phi must contain one angle per cross-section.")

    if yperiod is not None:
        # Repeat the first section one period later
        contours = contours + contours[:1]
        phi = np.append(phi, phi[0] + yperiod)
        pairs = [(k, k + 1) for k in range(len(contours) - 1)]
    elif close_toroidal:
        pairs = [(k, (k + 1) % len(contours)) for k in range(len(contours))]
    else:
        pairs = [(k, k + 1) for k in range(len(contours) - 1)]

    offsets = np.cumsum([0] + [len(c) for c in contours])
    vertices = np.vstack(
        [
            np.column_stack(rzphi_to_xyz(c[:, 0], c[:, 1], p))
            for c, p in zip(contours, phi)
        ]
    )

    triangles = []
    strip_rungs = []
    for (ia, ib), r in zip(pairs, tile_contours(contours, pairs)):
        a = offsets[ia] + r[:, 0]
        b = offsets[ib] + r[:, 1]
        strip_rungs.append(np.column_stack((a, b))[:-1])
        triangles.append(_strip_triangles(a, b))

    return vertices, np.vstack(triangles).astype(np.int64), strip_rungs


class Wall:
    """A closed wall surface, given by poloidal cross-sections

    Neighbouring cross-sections should be close enough that the flat
    triangles between them follow the wall, and should have a similar
    poloidal resolution, see the description of the module.

    Parameters
    ----------
    outer_lines : list of :py:obj:`~zoidberg.rzline.RZline`
        One closed contour per cross-section. The contours may have
        different numbers of points, start points and orientation.
    MXG : int, optional
        Unused, kept for backwards compatibility
    phi : array_like, optional
        Toroidal angle of each cross-section. If given, the mesh is
        created immediately, see `create_mesh`.
    close_toroidal, yperiod, check : optional
        Passed to `create_mesh`
    """

    def __init__(
        self,
        outer_lines,
        MXG=2,
        phi=None,
        close_toroidal=False,
        yperiod=None,
        check=True,
    ):
        self.contours = []
        for i, line in enumerate(outer_lines):
            try:
                self.contours.append(clean_contour(line.R, line.Z))
            except ValueError as error:
                raise ValueError(f"Wall cross-section {i}: {error}") from error

        if phi is not None:
            self.create_mesh(
                phi, close_toroidal=close_toroidal, yperiod=yperiod, check=check
            )

    def create_mesh(self, phi, close_toroidal=False, yperiod=None, check=True):
        """Create the triangle mesh

        Parameters
        ----------
        phi : array_like
            Toroidal angle of each cross-section, strictly increasing
        close_toroidal : bool
            The sections cover a full torus; same as yperiod=2*pi
        yperiod : float, optional
            Toroidal period of the wall. The sections must lie within
            one period, i.e. phi[-1] - phi[0] < yperiod. If neither
            yperiod nor close_toroidal is given the mesh is open at
            both ends, and field lines leave the wall there.
        check : bool
            Check the mesh, see `check`

        Neighbouring cross-sections, including the last and the first
        one period later, must be less than pi apart. A warning is
        given if the flat triangles between two cross-sections lie
        more than 5% of the size of the cross-sections inside the wall.
        """
        phi = np.asarray(phi, dtype=float)
        if phi.ndim != 1 or len(phi) != len(self.contours):
            raise ValueError("phi must contain one angle per cross-section")
        if len(phi) < 2 and yperiod is None and not close_toroidal:
            raise ValueError("An open wall needs at least two cross-sections")
        if np.any(np.diff(phi) <= 0):
            raise ValueError("phi must be strictly increasing")
        if close_toroidal and yperiod is None:
            yperiod = 2 * np.pi
        if yperiod is not None and phi[-1] - phi[0] >= yperiod:
            raise ValueError("The cross-sections must lie within one period")
        mesh_phi = phi if yperiod is None else np.append(phi, phi[0] + yperiod)
        spacing = np.diff(mesh_phi)
        if np.any(spacing >= np.pi):
            raise ValueError(
                "Neighbouring cross-sections must be less than pi apart in phi"
                f" (largest spacing {np.max(spacing)}); add more cross-sections"
            )
        self._warn_coarse(spacing)

        vertices, triangles, strip_rungs = create_wall_mesh(
            self.contours, phi, close_toroidal=False, yperiod=yperiod
        )

        self.vertices = vertices
        self.triangles = triangles
        self.strip_rungs = strip_rungs
        self.yperiod = yperiod
        # Toroidal angle of every section in the mesh, including the
        # repeated first section if periodic
        self.mesh_phi = mesh_phi
        self._intersect_cache = None

        tri = vertices[triangles]
        # With anticlockwise contours and increasing phi,
        # (p10 - p00) x (p01 - p00) points inwards, so reverse it to get
        # the outward normal
        normals = np.cross(tri[:, 2] - tri[:, 0], tri[:, 1] - tri[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        self.normals = normals / np.where(norm > 0, norm, 1.0)[:, None]

        if check:
            self.check()

    def _warn_coarse(self, spacing):
        """Warn if the triangles between sections k and k + 1 lie far
        inside the wall: a straight line between points at R and toroidal
        angles phi and phi + dphi comes R (1 - cos(dphi / 2)) closer to
        the axis"""
        n = len(self.contours)
        sag = np.empty(len(spacing))
        size = np.empty(len(spacing))
        for k, dphi in enumerate(spacing):
            A, B = self.contours[k], self.contours[(k + 1) % n]
            sag[k] = max(A[:, 0].max(), B[:, 0].max()) * (1 - np.cos(dphi / 2))
            size[k] = np.sqrt(
                min(_signed_area(A[:, 0], A[:, 1]), _signed_area(B[:, 0], B[:, 1]))
            )
        k = np.argmax(sag / size)
        if sag[k] > 0.05 * size[k]:
            warnings.warn(
                f"Cross-sections {k} and {k + 1} are {spacing[k]:.3g} rad apart:"
                f" the flat triangles between them lie up to {sag[k]:.3g}"
                f" ({sag[k] / size[k]:.0%} of the size of the cross-sections)"
                " inside the wall. Add more cross-sections.",
                stacklevel=3,
            )

    def check(self, fractions=_CHECK_FRACTIONS, min_area_ratio=0.5):
        """Check that the mesh is a valid wall surface

        - the rungs of every strip join each point of both contours in
          order, and the triangles are those of the rungs. Each strip is
          then a band of consistently oriented triangles.
        - no edge is shared by more than two triangles, which happens if
          a whole contour is joined to a single point, and no triangle
          is degenerate
        - the cross-sections at the given fractions of every strip are
          simple polygons, i.e. the wall does not fold over or twist,
          and they are anticlockwise with an area of at least
          min_area_ratio times the smaller area of the two neighbouring
          cross-sections, i.e. the wall is neither pinched nor turned
          inside out

        Folds narrower than the spacing of the fractions may be missed.
        This does not check whether the cross-sections are close enough
        for the triangles to follow the real wall.

        Raises
        ------
        ValueError
            Describing the first problem found
        """
        import shapely

        sizes = [len(c) for c in self.contours]
        if self.yperiod is not None:
            sizes.append(sizes[0])
        offsets = np.cumsum([0] + sizes)
        triangles = []
        for k, rungs in enumerate(self.strip_rungs):
            closed = np.vstack((rungs, rungs[:1]))
            a = closed[:, 0] - offsets[k]
            b = closed[:, 1] - offsets[k + 1]
            da = np.diff(a) % sizes[k]
            db = np.diff(b) % sizes[k + 1]
            if not (
                np.all((a >= 0) & (a < sizes[k]) & (b >= 0) & (b < sizes[k + 1]))
                and np.all((da + db == 1) & (da * db == 0))
                and np.sum(da) == sizes[k]
                and np.sum(db) == sizes[k + 1]
            ):
                raise ValueError(
                    f"Invalid rungs between cross-sections {k} and {k + 1}"
                )
            triangles.append(_strip_triangles(closed[:, 0], closed[:, 1]))
        if not np.array_equal(np.vstack(triangles), self.triangles):
            raise ValueError("The triangles of the wall mesh do not match its rungs")

        tri = self.vertices[self.triangles]
        area = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        scale = np.max(np.abs(self.vertices))
        degenerate = np.flatnonzero(area <= 1e-14 * scale**2)
        if degenerate.size:
            k = np.searchsorted(
                np.cumsum([len(r) for r in self.strip_rungs]), degenerate[0], "right"
            )
            raise ValueError(
                "The wall mesh contains degenerate triangles between"
                f" cross-sections {k} and {k + 1}"
            )

        # Triangles sharing an edge traverse it in opposite directions,
        # so a third triangle repeats one direction
        nv = len(self.vertices)
        start = self.triangles.ravel()
        end = self.triangles[:, [1, 2, 0]].ravel()
        directed = np.sort(start * nv + end)
        if np.any(directed[1:] == directed[:-1]):
            raise ValueError(
                "The wall mesh has edges shared by more than two triangles"
            )

        section_area = [_signed_area(*self.section(p)) for p in self.mesh_phi]
        for k in range(len(self.mesh_phi) - 1):
            phi0, phi1 = self.mesh_phi[k], self.mesh_phi[k + 1]
            for f in fractions:
                phi = phi0 + f * (phi1 - phi0)
                R, Z = self.section(phi)
                if not shapely.LinearRing(np.column_stack((R, Z))).is_simple:
                    raise ValueError(
                        f"The wall folds over between cross-sections {k} and {k + 1}"
                        f" (phi = {phi0} to {phi1}). Add more cross-sections,"
                        " or use check=False if this is intended"
                    )
                if _signed_area(R, Z) < min_area_ratio * min(
                    section_area[k], section_area[k + 1]
                ):
                    raise ValueError(
                        f"The wall is pinched between cross-sections {k} and {k + 1}"
                        f" (phi = {phi0} to {phi1}). Add more cross-sections,"
                        " or use check=False if this is intended"
                    )

    def to_base_period(self, phi):
        """Shift toroidal angles into the period covered by the mesh

        Returns
        -------
        phi_base, shift : ndarray
            phi_base = phi - shift, with phi_base in [mesh_phi[0], mesh_phi[0] + yperiod)
        """
        phi = np.asarray(phi, dtype=float)
        if self.yperiod is None:
            return phi, np.zeros_like(phi)
        phi0 = self.mesh_phi[0]
        shift = np.floor((phi - phi0) / self.yperiod) * self.yperiod
        return phi - shift, shift

    def section(self, phi):
        """Cross-section of the triangle mesh at toroidal angle phi

        This is the exact cut of the (faceted) mesh, so it is consistent
        with the intersection tests, also between the cross-sections
        the wall was built from.

        Parameters
        ----------
        phi : float

        Returns
        -------
        R, Z : ndarray
            Polygon, anticlockwise in (R, Z), closed implicitly: the
            first point is not repeated at the end. May contain repeated
            points.
        """
        phi_b, _ = self.to_base_period(phi)
        mesh_phi = self.mesh_phi
        tol = 1e-12
        if phi_b < mesh_phi[0] - tol or phi_b > mesh_phi[-1] + tol:
            raise ValueError(f"phi={phi} is outside the wall")

        j = np.clip(
            np.searchsorted(mesh_phi, phi_b, side="right") - 1, 0, len(mesh_phi) - 2
        )
        # Every triangle of the strip has one rung on each side, so the
        # cuts of the rungs in order form the section. The plane through
        # the axis is cut only at phi_b, not phi_b + pi, as the sections
        # are less than pi apart
        rungs = self.strip_rungs[j]
        A = self.vertices[rungs[:, 0]]
        B = self.vertices[rungs[:, 1]]

        m = np.array([-np.sin(phi_b), np.cos(phi_b), 0.0])
        mA = A @ m
        mB = B @ m
        denom = mA - mB
        lam = np.divide(mA, denom, out=np.zeros_like(mA), where=np.abs(denom) > 0)
        lam = np.clip(lam, 0.0, 1.0)
        P = A + lam[:, None] * (B - A)
        R = P[:, 0] * np.cos(phi_b) + P[:, 1] * np.sin(phi_b)
        return R, P[:, 2]

    def plot_mesh(self):
        import pyvista as pv

        # Convert triangular connectivity to PyVista format.
        # Each triangle is represented as: [3, vertex0, vertex1, vertex2]
        faces = np.column_stack(
            (np.full(len(self.triangles), 3), self.triangles)
        ).ravel()

        mesh = pv.PolyData(self.vertices, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="cornflowerblue", show_edges=True, line_width=0.3)

        plotter.add_axes()
        plotter.show_grid()
        plotter.show()


def _wall_cache(wall):
    """Precompute and cache triangle geometry + bounding boxes on the wall object."""
    cache = getattr(wall, "_intersect_cache", None)
    if cache is not None:
        return cache

    vertices = wall.vertices
    triangles = wall.triangles

    tri = vertices[triangles]
    v0 = tri[:, 0]
    edge1 = tri[:, 1] - v0
    edge2 = tri[:, 2] - v0

    tri_min = tri.min(axis=1)
    tri_max = tri.max(axis=1)

    centroid = tri.mean(axis=1)
    radius = np.max(np.linalg.norm(tri - centroid[:, None, :], axis=-1), axis=1)

    cache = {
        "v0": v0,
        "edge1": edge1,
        "edge2": edge2,
        "tri_min": tri_min,
        "tri_max": tri_max,
        "centroid": centroid,
        "radius": radius,
    }
    wall._intersect_cache = cache
    return cache


def first_wall_intersection_fast(wall, line, eps=1e-10):
    """
    Find the first intersection of a traced 3D polyline with the wall.

    Returns
    -------
    point, segment_id, triangle_id, distance
    """

    line = np.asarray(line, dtype=float)

    cache = _wall_cache(wall)
    v0, edge1, edge2 = cache["v0"], cache["edge1"], cache["edge2"]
    tri_min, tri_max = cache["tri_min"], cache["tri_max"]

    segment_vectors = np.diff(line, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    # ---- Global broad phase: discard triangles nowhere near the whole line ----
    line_min = line.min(axis=0) - eps
    line_max = line.max(axis=0) + eps

    global_mask = np.all((tri_max >= line_min) & (tri_min <= line_max), axis=1)
    cand = np.flatnonzero(global_mask)

    if cand.size == 0:
        return None, None, None, None

    g_v0, g_e1, g_e2 = v0[cand], edge1[cand], edge2[cand]
    g_tmin, g_tmax = tri_min[cand], tri_max[cand]

    for i in range(len(line) - 1):
        seg_len = segment_lengths[i]
        if seg_len < eps:
            continue

        origin = line[i]
        end = line[i + 1]
        direction = end - origin

        # ---- Per-segment broad phase on the already-reduced candidate set ----
        seg_min = np.minimum(origin, end) - eps
        seg_max = np.maximum(origin, end) + eps

        local_mask = np.all((g_tmax >= seg_min) & (g_tmin <= seg_max), axis=1)
        idx = np.flatnonzero(local_mask)
        if idx.size == 0:
            continue

        e1 = g_e1[idx]
        e2 = g_e2[idx]
        p0 = g_v0[idx]

        h = np.cross(direction, e2)
        a = np.einsum("ij,ij->i", e1, h)

        # a scales with the cube of the size, so compare it relative to that
        scale = seg_len * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        valid = np.abs(a) > eps * scale
        if not np.any(valid):
            continue

        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]

        s = origin - p0
        u = f * np.einsum("ij,ij->i", s, h)

        q = np.cross(s, e1)
        v = f * np.einsum("ij,j->i", q, direction)
        t = f * np.einsum("ij,ij->i", e2, q)

        hit = (
            valid
            & (u >= -eps)
            & (v >= -eps)
            & (u + v <= 1.0 + eps)
            & (t >= -eps)
            & (t <= 1.0 + eps)
        )

        hit_local = np.flatnonzero(hit)
        if hit_local.size > 0:
            best = hit_local[np.argmin(t[hit_local])]
            j = cand[idx[best]]
            tj = t[best]

            point = origin + tj * direction
            distance = cumulative_lengths[i] + tj * seg_len

            return point, i, j, distance

    return None, None, None, None
