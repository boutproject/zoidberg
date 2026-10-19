import numpy as np

from .grid import Grid

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import gmsh

def rzphi_to_xyz(r,z,phi):
    return r*np.cos(phi), r*np.sin(phi), z

def align_contours(A):
    """
    Align ordered contours by finding the optimal cyclic shift
    between neighboring cross-sections.

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
    shifts : ndarray
        Cyclic shift applied to each cross-section.
    """
    B = np.array(A, copy=True)

    n_sections, n_points, _ = B.shape
    shifts = np.zeros(n_sections, dtype=int)

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
        shifts[j] = (shifts[j - 1] + k_best) % n_points

    return B

def create_wall_mesh(B, phi, close_toroidal=True):
    """
    Create a triangular surface mesh in Gmsh from aligned R-Z contours.

    Parameters
    ----------
    B : ndarray, shape (n_sections, n_points, 2)
        B[j, i, 0] = R
        B[j, i, 1] = Z

    phi : ndarray, shape (n_sections,)
        Toroidal angles in radians.

    filename : str
        Output Gmsh mesh filename.

    close_toroidal : bool
        Connect the last cross-section to the first.
        Use True only when the sections cover a full 2*pi torus.
    """

    B = np.asarray(B, dtype=float)
    phi = np.asarray(phi, dtype=float)

    n_sections, n_points, _ = B.shape

    if len(phi) != n_sections:
        raise ValueError("phi must contain one angle per cross-section.")

    # --------------------------------------------------
    # 1. Convert R-Z contours to Cartesian coordinates
    # --------------------------------------------------

    vertices = np.zeros((n_sections, n_points, 3))

    vertices[:, :, 0] = B[:, :, 0] * np.cos(phi[:, None])
    vertices[:, :, 1] = B[:, :, 0] * np.sin(phi[:, None])
    vertices[:, :, 2] = B[:, :, 1]

    # Flatten to (n_sections * n_points, 3)
    vertices = vertices.reshape(-1, 3)

    def idx(j, i):
        return j * n_points + (i % n_points)

    # --------------------------------------------------
    # 2. Connect neighboring cross-sections
    # --------------------------------------------------

    triangles = []

    n_connections = n_sections if close_toroidal else n_sections - 1

    for j in range(n_connections):

        j_next = (j + 1) % n_sections

        for i in range(n_points):

            i_next = (i + 1) % n_points

            p00 = idx(j, i)
            p10 = idx(j, i_next)
            p01 = idx(j_next, i)
            p11 = idx(j_next, i_next)

            # Two triangles per quadrilateral
            triangles.append([p00, p10, p01])
            triangles.append([p10, p11, p01])

    triangles = np.asarray(triangles, dtype=np.int64)

    return vertices, triangles


class Wall(object):

    def __init__(self, outer_lines, MXG = 2):
        
        # Number of cross sections
        n_crosssection = len(outer_lines)
        
        # Number of points, basically nz
        n_pnts = outer_lines[0].R.size
        
        #Create a 3D array that contains all the grid points
        # 1st dimension is the cross section
        # 2nd dimension is the different points        
        # 3rd dimension is the coordinates of the point, so 2
        pnts = np.zeros((n_crosssection, n_pnts, 2))
        
        for i in range(n_crosssection):
            pnts[i,:,0] = outer_lines[i].R
            pnts[i,:,1] = outer_lines[i].Z
        
        self.pnts = np.copy(pnts)
        
        # This assumes that all the points are ordered
        
        ordered_pnts = align_contours(pnts)
        self.ordered_pnts = np.copy(ordered_pnts)
        
    def create_mesh(self, phi, close_toroidal):
        vertices, triangles = create_wall_mesh(
            self.ordered_pnts,
            phi,
            close_toroidal=close_toroidal
        )
        
        self.vertices = vertices
        self.triangles = triangles
        
    def plot_mesh(self):
        import pyvista as pv


        # Convert triangular connectivity to PyVista format.
        # Each triangle is represented as: [3, vertex0, vertex1, vertex2]
        faces = np.column_stack((
            np.full(len(self.triangles), 3),
            self.triangles
        )).ravel()

        mesh = pv.PolyData(self.vertices, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,
            color="cornflowerblue",
            show_edges=True,
            line_width=0.3
        )



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

    cache = dict(v0=v0, edge1=edge1, edge2=edge2,
                 tri_min=tri_min, tri_max=tri_max)
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
