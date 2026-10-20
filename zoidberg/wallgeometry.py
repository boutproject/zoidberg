"""Triangulated 3D wall built from poloidal cross-sections of the wall

The wall is given as a set of closed (R, Z) contours at a list of
toroidal angles. Neighbouring contours are joined into a triangle mesh
in Cartesian (X, Y, Z) coordinates, so that all intersection tests are
done in real space.

The wall resolution is independent of the grid: typically the wall is
built from many more cross-sections than the grid has poloidal slices.

"""

import numpy as np


def rzphi_to_xyz(r, z, phi):
    return r * np.cos(phi), r * np.sin(phi), z


def _signed_area(r, z):
    """Signed area of closed (r, z) polygons along the last axis.
    Positive if anticlockwise in the (R, Z) plane"""
    return 0.5 * np.sum(
        r * np.roll(z, -1, axis=-1) - np.roll(r, -1, axis=-1) * z, axis=-1
    )


def orient_contours(A):
    """
    Make all contours run anticlockwise in the (R, Z) plane.

    Parameters
    ----------
    A : ndarray, shape (n_sections, n_points, 2)

    Returns
    -------
    B : ndarray
        Copy of A with clockwise contours reversed.
    """
    B = np.array(A, copy=True)
    clockwise = _signed_area(B[:, :, 0], B[:, :, 1]) < 0
    B[clockwise] = B[clockwise, ::-1]
    return B


def align_contours(A):
    """
    Align ordered contours by finding the optimal cyclic shift
    between neighboring cross-sections.

    All contours must have the same orientation, see `orient_contours`.

    Parameters
    ----------
    A : ndarray, shape (n_sections, n_points, 2)
        A[j, i, 0] = R coordinate
        A[j, i, 1] = Z coordinate

    Returns
    -------
    B : ndarray
        Copy of A with contours cyclically shifted to align
        neighboring cross-sections.
    """
    B = np.array(A, copy=True)

    n_sections, n_points, _ = B.shape

    for j in range(1, n_sections):
        reference = B[j - 1]
        current = B[j]

        # Find the best cyclic shift.
        errors = np.empty(n_points)

        for k in range(n_points):
            shifted = np.roll(current, -k, axis=0)

            diff = reference - shifted

            errors[k] = np.sum(diff**2)

        # Apply the optimal shift.
        k_best = np.argmin(errors)

        B[j] = np.roll(current, -k_best, axis=0)

    return B


def create_wall_mesh(B, phi, close_toroidal=True, yperiod=None):
    """
    Create a triangular surface mesh from aligned R-Z contours.

    Parameters
    ----------
    B : ndarray, shape (n_sections, n_points, 2)
        B[j, i, 0] = R
        B[j, i, 1] = Z

    phi : ndarray, shape (n_sections,)
        Toroidal angles in radians.

    close_toroidal : bool
        Connect the last cross-section to the first.
        Use True only when the sections cover a full 2*pi torus.

    yperiod : float, optional
        Toroidal period of the wall, e.g. 2*pi/5 for a five-period
        stellarator. If given, the first cross-section is repeated at
        phi[0] + yperiod and connected to the last one, so that the mesh
        covers exactly one period. Overrides close_toroidal.

    Returns
    -------
    vertices : ndarray, shape (n_vertices, 3)
        Cartesian coordinates of the vertices
    triangles : ndarray, shape (n_triangles, 3)
        Vertex indices of each triangle. The two triangles between
        sections j, j+1 and points i, i+1 have indices
        2 * (j * n_points + i) and 2 * (j * n_points + i) + 1
    """

    B = np.asarray(B, dtype=float)
    phi = np.asarray(phi, dtype=float)

    n_sections, n_points, _ = B.shape

    if len(phi) != n_sections:
        raise ValueError("phi must contain one angle per cross-section.")

    if yperiod is not None:
        # Repeat the first section one period later
        B = np.concatenate((B, B[:1]))
        phi = np.append(phi, phi[0] + yperiod)
        n_sections += 1
        close_toroidal = False

    # --------------------------------------------------
    # 1. Convert R-Z contours to Cartesian coordinates
    # --------------------------------------------------

    vertices = np.zeros((n_sections, n_points, 3))

    vertices[:, :, 0] = B[:, :, 0] * np.cos(phi[:, None])
    vertices[:, :, 1] = B[:, :, 0] * np.sin(phi[:, None])
    vertices[:, :, 2] = B[:, :, 1]

    # Flatten to (n_sections * n_points, 3)
    vertices = vertices.reshape(-1, 3)

    # --------------------------------------------------
    # 2. Connect neighboring cross-sections
    # --------------------------------------------------

    n_connections = n_sections if close_toroidal else n_sections - 1

    j, i = np.meshgrid(np.arange(n_connections), np.arange(n_points), indexing="ij")
    j_next = (j + 1) % n_sections
    i_next = (i + 1) % n_points

    p00 = j * n_points + i
    p10 = j * n_points + i_next
    p01 = j_next * n_points + i
    p11 = j_next * n_points + i_next

    # Two triangles per quadrilateral
    triangles = np.stack(
        (
            np.stack((p00, p10, p01), axis=-1),
            np.stack((p10, p11, p01), axis=-1),
        ),
        axis=2,
    ).reshape(-1, 3)

    return vertices, triangles.astype(np.int64)


class Wall:
    """A closed wall surface, given by poloidal cross-sections

    Parameters
    ----------
    outer_lines : list of :py:obj:`~zoidberg.rzline.RZline`
        One closed contour per cross-section. All contours must have
        the same number of points.
    MXG : int, optional
        Unused, kept for backwards compatibility
    phi : array_like, optional
        Toroidal angle of each cross-section. If given, the mesh is
        created immediately, see `create_mesh`.
    close_toroidal, yperiod : optional
        Passed to `create_mesh`
    """

    def __init__(
        self, outer_lines, MXG=2, phi=None, close_toroidal=False, yperiod=None
    ):
        # Number of cross sections
        n_crosssection = len(outer_lines)

        # Number of points, basically nz
        n_pnts = outer_lines[0].R.size

        # Create a 3D array that contains all the grid points
        # 1st dimension is the cross section
        # 2nd dimension is the different points
        # 3rd dimension is the coordinates of the point, so 2
        pnts = np.zeros((n_crosssection, n_pnts, 2))

        for i in range(n_crosssection):
            if outer_lines[i].R.size != n_pnts:
                raise ValueError(
                    "All wall contours must have the same number of points"
                )
            pnts[i, :, 0] = outer_lines[i].R
            pnts[i, :, 1] = outer_lines[i].Z

        self.pnts = np.copy(pnts)

        # This assumes that all the points are ordered
        ordered_pnts = align_contours(orient_contours(pnts))
        self.ordered_pnts = np.copy(ordered_pnts)

        if phi is not None:
            self.create_mesh(phi, close_toroidal=close_toroidal, yperiod=yperiod)

    def create_mesh(self, phi, close_toroidal=False, yperiod=None):
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
            both ends.
        """
        phi = np.asarray(phi, dtype=float)
        if np.any(np.diff(phi) <= 0):
            raise ValueError("phi must be strictly increasing")
        if close_toroidal and yperiod is None:
            yperiod = 2 * np.pi
        if yperiod is not None and phi[-1] - phi[0] >= yperiod:
            raise ValueError("The cross-sections must lie within one period")

        vertices, triangles = create_wall_mesh(
            self.ordered_pnts, phi, close_toroidal=False, yperiod=yperiod
        )

        self.vertices = vertices
        self.triangles = triangles
        self.yperiod = yperiod
        # Toroidal angle of every section in the mesh, including the
        # repeated first section if periodic
        self.mesh_phi = phi if yperiod is None else np.append(phi, phi[0] + yperiod)
        self.n_points = self.ordered_pnts.shape[1]
        self._intersect_cache = None

        tri = vertices[triangles]
        # With anticlockwise contours and increasing phi,
        # (p10 - p00) x (p01 - p00) points inwards, so reverse it to get
        # the outward normal
        normals = np.cross(tri[:, 2] - tri[:, 0], tri[:, 1] - tri[:, 0])
        self.normals = normals / np.linalg.norm(normals, axis=1)[:, None]

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

        Returns
        -------
        R, Z : ndarray
            Closed polygon, anticlockwise in (R, Z). May contain
            repeated points.
        """
        phi_b, _ = self.to_base_period(phi)
        mesh_phi = self.mesh_phi
        tol = 1e-12
        if phi_b < mesh_phi[0] - tol or phi_b > mesh_phi[-1] + tol:
            raise ValueError(f"phi={phi} is outside the wall")

        j = np.clip(
            np.searchsorted(mesh_phi, phi_b, side="right") - 1, 0, len(mesh_phi) - 2
        )
        n = self.n_points
        V = self.vertices.reshape(-1, n, 3)

        m = np.array([-np.sin(phi_b), np.cos(phi_b), 0.0])

        def cut(A, B):
            mA = A @ m
            mB = B @ m
            denom = mA - mB
            lam = np.divide(mA, denom, out=np.zeros_like(mA), where=np.abs(denom) > 0)
            lam = np.clip(lam, 0.0, 1.0)
            return A + lam[:, None] * (B - A)

        # Edges between the sections, and the diagonals of each quad
        E = cut(V[j], V[j + 1])
        D = cut(np.roll(V[j], -1, axis=0), V[j + 1])

        P = np.stack((E, D), axis=1).reshape(-1, 3)
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

        valid = np.abs(a) > eps
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
