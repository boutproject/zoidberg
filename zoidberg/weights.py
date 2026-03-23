"""Construct cell-space numbering, leg metadata, and leg operators.

This module assigns global numbers for the cell space ``C`` and constructs
metadata and operators for directional leg spaces:

- ``L+``: forward legs
- ``L-``: backward legs

The cell space ``C`` contains:

1. evolving interior cells
2. inner radial X-boundary cells
3. outer radial X-boundary cells
4. backward parallel boundary cells
5. forward parallel boundary cells

The forward and backward leg spaces contain rows only for evolving interior
source cells. Each interior cell contributes either one or two rows in each leg
space:

- one row if all sub-samples map to interior cells, or all map to the boundary
- two rows if some sub-samples map to interior cells and others hit the
  boundary

For each direction this module exports:

- explicit leg-number metadata arrays
- a leg-weight vector containing the quadrature weight of each leg row
- a destination operator mapping from cell space ``C`` to leg space
  (rows = leg rows, columns = cell indices)

No injection matrices are generated. The consumer can reconstruct local
cell-to-leg injection operators directly from the leg-number metadata.

Sparse storage format
---------------------
Sparse matrices are exported in standard CSR (Compressed Sparse Row) format:

- ``weights[nnz]``: nonzero matrix entries
- ``columns[nnz]``: column indices corresponding to ``weights``
- ``rows[n_rows + 1]``: CSR row pointer array

Row ``i`` occupies entries ``weights[rows[i]:rows[i+1]]`` and
``columns[rows[i]:rows[i+1]]``.
"""

from __future__ import annotations

import numpy as np

FORWARD_DIRECTION = {
    "name": "forward",
    "xt_key": "forward_xt_primes",
    "xt_fallback": "forward_xt_prime",
    "zt_key": "forward_zt_primes",
    "zt_fallback": "forward_zt_prime",
    "boundary_number_key": "forward_boundary_number",
    "y_offset": +1,
}

BACKWARD_DIRECTION = {
    "name": "backward",
    "xt_key": "backward_xt_primes",
    "xt_fallback": "backward_xt_prime",
    "zt_key": "backward_zt_primes",
    "zt_fallback": "backward_zt_prime",
    "boundary_number_key": "backward_boundary_number",
    "y_offset": -1,
}


def _get_map_array(maps, primary_key, fallback_key=None):
    """Return an array from the map dictionary."""
    if primary_key in maps:
        return maps[primary_key]
    if fallback_key is not None and fallback_key in maps:
        return maps[fallback_key]
    if fallback_key is None:
        raise KeyError(f"Missing required map array '{primary_key}'")
    raise KeyError(
        f"Missing required map array '{primary_key}' "
        f"(fallback '{fallback_key}' also not found)"
    )


def _as_subsample_array(array):
    """Ensure mapped coordinate arrays have shape ``(nx, ny, nz, n_subsamples)``."""
    arr = np.asarray(array)
    if arr.ndim == 3:
        return arr[..., np.newaxis]
    if arr.ndim == 4:
        return arr
    raise ValueError(
        f"Expected mapped coordinate array with 3 or 4 dimensions, got shape {arr.shape}"
    )


def _get_subcell_weights(maps, n_subsamples):
    """Return the quadrature weights for sub-samples as a 1D array."""
    if "subcell_weights" in maps:
        weights = np.asarray(maps["subcell_weights"], dtype=float)
    elif n_subsamples == 1:
        weights = np.full(1, 1.0, dtype=float)
    else:
        raise ValueError("Missing subcell_weights")

    if weights.ndim != 1:
        raise ValueError(
            f"subcell_weights must be one-dimensional, got shape {weights.shape}"
        )
    if len(weights) != n_subsamples:
        raise ValueError(
            f"subcell_weights has length {len(weights)} but expected {n_subsamples}"
        )
    if abs(np.sum(weights) - 1.0) >= 1e-8:
        raise ValueError(f"subcell_weights should sum to 1, got {np.sum(weights)}")
    return weights


def hits_radial_boundary(x_mapped, nx, mxg):
    """Return True if a mapped X coordinate lies in the radial boundary region."""
    return (x_mapped < mxg) or (x_mapped > nx - mxg - 1)


def classify_leg_contributions(mapped_xs, subcell_weights, nx, mxg):
    """Classify one source cell's sub-samples into interior and boundary parts.

    Parameters
    ----------
    mapped_xs : array-like
        Forward or backward mapped X coordinates for one source cell.
    subcell_weights : array-like
        Quadrature weights for the sub-samples.
    nx : int
        Number of X points.
    mxg : int
        Width of the radial boundary region.

    Returns
    -------
    dict
        Classification information for one source cell.
    """
    mapped_xs = np.asarray(mapped_xs, dtype=float)
    subcell_weights = np.asarray(subcell_weights, dtype=float)

    hits_boundary_mask = np.fromiter(
        (hits_radial_boundary(x, nx, mxg) for x in mapped_xs),
        dtype=bool,
        count=len(mapped_xs),
    )

    boundary_fraction = float(np.sum(subcell_weights[hits_boundary_mask]))
    interior_fraction = float(np.sum(subcell_weights[~hits_boundary_mask]))

    return {
        "hits_boundary": hits_boundary_mask,
        "has_boundary_leg": np.any(hits_boundary_mask),
        "has_interior_leg": np.any(~hits_boundary_mask),
        "boundary_fraction": boundary_fraction,
        "interior_fraction": interior_fraction,
    }


def assign_cell_space_numbers(maps):
    """Assign global numbers to cell-space interior and boundary cells.

    Parameters
    ----------
    maps : dict
        Field-line map dictionary.

    Returns
    -------
    dict
        Cell-space numbering metadata.
    """
    nx, ny, nz = maps["R"].shape
    mxg = maps["MXG"]

    n_evolving = (nx - 2 * mxg) * ny * nz
    cell_number = np.full((nx, ny, nz), -1, dtype=int)

    next_cell = 0

    # Interior cells
    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    # Inner X boundary
    for x in range(mxg):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    # Outer X boundary
    for x in range(nx - mxg, nx):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    forward_xts = _as_subsample_array(
        _get_map_array(maps, "forward_xt_primes", "forward_xt_prime")
    )
    backward_xts = _as_subsample_array(
        _get_map_array(maps, "backward_xt_primes", "backward_xt_prime")
    )

    backward_boundary_number = np.full((nx, ny, nz), -1, dtype=int)
    forward_boundary_number = np.full((nx, ny, nz), -1, dtype=int)

    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                if np.any(
                    [hits_radial_boundary(v, nx, mxg) for v in backward_xts[x, y, z, :]]
                ):
                    backward_boundary_number[x, y, z] = next_cell
                    next_cell += 1
                if np.any(
                    [hits_radial_boundary(v, nx, mxg) for v in forward_xts[x, y, z, :]]
                ):
                    forward_boundary_number[x, y, z] = next_cell
                    next_cell += 1

    return {
        "N_cells": next_cell,
        "N_evolving": n_evolving,
        "cell_number": cell_number,
        "forward_boundary_number": forward_boundary_number,
        "backward_boundary_number": backward_boundary_number,
    }


def catmull_rom_weights(u):
    """Return 1D Catmull-Rom interpolation weights for fractional offset ``u``."""
    return 0.5 * np.array(
        [
            -(u**3) + 2.0 * u**2 - u,
            3.0 * u**3 - 5.0 * u**2 + 2.0,
            -3.0 * u**3 + 4.0 * u**2 + u,
            u**3 - u**2,
        ]
    )


def shift_weights_out_of_x_boundaries(weights_x, x_base, x_offsets, nx, mxg):
    """Move X interpolation weight off radial boundary points."""
    adjusted = np.array(weights_x, copy=True)

    for wi, offset in enumerate(x_offsets):
        if x_base + offset < mxg:
            adjusted[wi + 1] += adjusted[wi]
            adjusted[wi] = 0.0

    for wi in reversed(range(len(x_offsets))):
        offset = x_offsets[wi]
        if x_base + offset >= nx - mxg:
            adjusted[wi - 1] += adjusted[wi]
            adjusted[wi] = 0.0

    if abs(np.sum(adjusted) - 1.0) >= 1e-8:
        raise ValueError(f"Adjusted X weights should sum to 1, got {np.sum(adjusted)}")

    return adjusted


class SparseRowMatrixBuilder:
    """Accumulate sparse rows before exporting to standard CSR format."""

    def __init__(self, n_rows):
        self.rows = [[] for _ in range(n_rows)]

    def set_row(self, row, entries):
        """Assign a row exactly once."""
        if self.rows[row]:
            raise ValueError(f"Row {row} assigned more than once")
        self.rows[row] = list(entries)

    def add_to_entry(self, row, col, weight):
        """Add to a matrix entry, accumulating duplicate columns in a row."""
        for i, (entry_col, entry_weight) in enumerate(self.rows[row]):
            if entry_col == col:
                self.rows[row][i] = (col, entry_weight + weight)
                return
        self.rows[row].append((col, weight))

    def to_csr(self):
        """Export matrix to standard CSR format using preallocated arrays."""
        n_rows = len(self.rows)
        nnz_per_row = np.fromiter((len(r) for r in self.rows), dtype=int, count=n_rows)
        row_ptr = np.empty(n_rows + 1, dtype=int)
        row_ptr[0] = 0
        np.cumsum(nnz_per_row, out=row_ptr[1:])

        nnz = row_ptr[-1]
        weights = np.empty(nnz, dtype=float)
        columns = np.empty(nnz, dtype=int)

        for row, entries in enumerate(self.rows):
            start = row_ptr[row]
            for i, (col, weight) in enumerate(entries):
                columns[start + i] = col
                weights[start + i] = weight

        return {"weights": weights, "columns": columns, "rows": row_ptr}


def build_leg_space_metadata(cell_space, maps, direction):
    """Build leg numbering metadata for one direction.

    Parameters
    ----------
    cell_space : dict
        Output of :func:`assign_cell_space_numbers`.
    maps : dict
        Field-line map dictionary.
    direction : dict
        Direction configuration, e.g. ``FORWARD_DIRECTION``.

    Returns
    -------
    dict
        Leg-space numbering metadata for the chosen direction.
    """
    cell_number = cell_space["cell_number"]
    nx, ny, nz = cell_number.shape
    mxg = maps["MXG"]

    xt_primes = _as_subsample_array(
        _get_map_array(maps, direction["xt_key"], direction["xt_fallback"])
    )
    n_subsamples = xt_primes.shape[3]
    subcell_weights = _get_subcell_weights(maps, n_subsamples)

    interior_leg_number = np.full((nx, ny, nz), -1, dtype=int)
    boundary_leg_number = np.full((nx, ny, nz), -1, dtype=int)
    interior_fraction = np.zeros((nx, ny, nz), dtype=float)
    boundary_fraction = np.zeros((nx, ny, nz), dtype=float)
    has_interior_leg = np.zeros((nx, ny, nz), dtype=bool)
    has_boundary_leg = np.zeros((nx, ny, nz), dtype=bool)

    next_leg = 0
    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                classification = classify_leg_contributions(
                    xt_primes[x, y, z, :], subcell_weights, nx, mxg
                )

                has_interior_leg[x, y, z] = classification["has_interior_leg"]
                has_boundary_leg[x, y, z] = classification["has_boundary_leg"]
                interior_fraction[x, y, z] = classification["interior_fraction"]
                boundary_fraction[x, y, z] = classification["boundary_fraction"]

                if classification["has_interior_leg"]:
                    interior_leg_number[x, y, z] = next_leg
                    next_leg += 1
                if classification["has_boundary_leg"]:
                    boundary_leg_number[x, y, z] = next_leg
                    next_leg += 1

    return {
        "direction_name": direction["name"],
        "n_legs": next_leg,
        "interior_leg_number": interior_leg_number,
        "boundary_leg_number": boundary_leg_number,
        "interior_fraction": interior_fraction,
        "boundary_fraction": boundary_fraction,
        "has_interior_leg": has_interior_leg,
        "has_boundary_leg": has_boundary_leg,
    }


def build_leg_weight_vector(leg_space):
    """Build the leg-weight vector for one leg space.

    Parameters
    ----------
    leg_space : dict
        Output of :func:`build_leg_space_metadata`.

    Returns
    -------
    numpy.ndarray
        One weight per leg row.
    """
    weights = np.zeros(leg_space["n_legs"], dtype=float)

    interior_leg_number = leg_space["interior_leg_number"]
    boundary_leg_number = leg_space["boundary_leg_number"]
    interior_fraction = leg_space["interior_fraction"]
    boundary_fraction = leg_space["boundary_fraction"]
    has_interior_leg = leg_space["has_interior_leg"]
    has_boundary_leg = leg_space["has_boundary_leg"]

    nx, ny, nz = interior_leg_number.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not (has_interior_leg[x, y, z] or has_boundary_leg[x, y, z]):
                    continue

                split = has_interior_leg[x, y, z] and has_boundary_leg[x, y, z]

                if has_interior_leg[x, y, z]:
                    row = interior_leg_number[x, y, z]
                    weights[row] = interior_fraction[x, y, z] if split else 1.0
                if has_boundary_leg[x, y, z]:
                    row = boundary_leg_number[x, y, z]
                    weights[row] = boundary_fraction[x, y, z] if split else 1.0

    return weights


def accumulate_interior_stencil(
    matrix_builder,
    row,
    mapped_xs,
    mapped_zs,
    subcell_weights,
    hits_boundary_mask,
    y,
    y_offset,
    cell_number,
    mxg,
):
    """Accumulate the normalized interior stencil for one leg row."""
    nx, ny, nz = cell_number.shape
    x_offsets = [-1, 0, 1, 2]

    interior_weight = float(np.sum(subcell_weights[~hits_boundary_mask]))
    if interior_weight <= 0.0:
        raise ValueError("Cannot build interior leg row with zero interior weight")

    for mapped_x, mapped_z, subcell_weight, hit_boundary in zip(
        mapped_xs, mapped_zs, subcell_weights, hits_boundary_mask
    ):
        if hit_boundary:
            continue

        x_base = int(mapped_x)
        z_base = int(mapped_z)

        weights_x = catmull_rom_weights(mapped_x - x_base)
        weights_z = catmull_rom_weights(mapped_z - z_base)
        weights_x = shift_weights_out_of_x_boundaries(
            weights_x, x_base, x_offsets, nx, mxg
        )

        if abs(np.sum(weights_z) - 1.0) >= 1e-8:
            raise ValueError(f"Z weights should sum to 1, got {np.sum(weights_z)}")

        scale = subcell_weight / interior_weight

        for x_offset, x_weight in zip(x_offsets, weights_x):
            if abs(x_weight) < 1e-10:
                continue
            for z_offset, z_weight in zip(x_offsets, weights_z):
                col = cell_number[
                    np.clip(x_base + x_offset, 0, nx - 1),
                    (y + y_offset + ny) % ny,
                    (z_base + z_offset + nz) % nz,
                ]
                if col < 0:
                    raise ValueError(
                        "Invalid column index at "
                        f"x={x_base + x_offset}, y={(y + y_offset + ny) % ny}, "
                        f"z={(z_base + z_offset + nz) % nz}"
                    )
                matrix_builder.add_to_entry(row, col, scale * x_weight * z_weight)


def build_leg_destination_operator(cell_space, leg_space, maps, direction):
    """Build the destination operator for one direction.

    The operator has rows in the chosen leg space and columns in cell space.
    Boundary leg rows map to one boundary cell in ``C``. Interior leg rows map
    to a normalized interpolation stencil over interior/X-boundary cells in
    ``C``.
    """
    cell_number = cell_space["cell_number"]
    boundary_number = cell_space[direction["boundary_number_key"]]
    nx, ny, nz = cell_number.shape
    mxg = maps["MXG"]

    xt_primes = _as_subsample_array(
        _get_map_array(maps, direction["xt_key"], direction["xt_fallback"])
    )
    zt_primes = _as_subsample_array(
        _get_map_array(maps, direction["zt_key"], direction["zt_fallback"])
    )
    subcell_weights = _get_subcell_weights(maps, xt_primes.shape[3])

    builder = SparseRowMatrixBuilder(leg_space["n_legs"])

    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                mapped_xs = xt_primes[x, y, z, :]
                mapped_zs = zt_primes[x, y, z, :]
                classification = classify_leg_contributions(
                    mapped_xs, subcell_weights, nx, mxg
                )

                if leg_space["has_boundary_leg"][x, y, z]:
                    row = leg_space["boundary_leg_number"][x, y, z]
                    col = boundary_number[x, y, z]
                    if col < 0:
                        raise ValueError(
                            f"Expected valid {direction['name']} boundary number at x={x}, y={y}, z={z}"
                        )
                    builder.set_row(row, [(col, 1.0)])

                if leg_space["has_interior_leg"][x, y, z]:
                    row = leg_space["interior_leg_number"][x, y, z]
                    accumulate_interior_stencil(
                        matrix_builder=builder,
                        row=row,
                        mapped_xs=mapped_xs,
                        mapped_zs=mapped_zs,
                        subcell_weights=subcell_weights,
                        hits_boundary_mask=classification["hits_boundary"],
                        y=y,
                        y_offset=direction["y_offset"],
                        cell_number=cell_number,
                        mxg=mxg,
                    )

    return builder.to_csr()


def calculate_parallel_map_operators(maps):
    """Calculate cell numbering, leg metadata, leg weights,
    and forward/backward operators.

    Parameters
    ----------
    maps : dict
        Field-line map dictionary.

    Returns
    -------
    dict
        Dictionary containing cell-space numbering, forward/backward leg
        numbering metadata, leg-weight vectors, and the CSR arrays for
        ``forward`` and ``backward``.
    """
    cell_space = assign_cell_space_numbers(maps)

    forward_leg_space = build_leg_space_metadata(cell_space, maps, FORWARD_DIRECTION)
    backward_leg_space = build_leg_space_metadata(cell_space, maps, BACKWARD_DIRECTION)

    forward_operator = build_leg_destination_operator(
        cell_space, forward_leg_space, maps, FORWARD_DIRECTION
    )
    backward_operator = build_leg_destination_operator(
        cell_space, backward_leg_space, maps, BACKWARD_DIRECTION
    )

    return {
        # Cell numbering
        "total_cells": cell_space["N_cells"],
        "cell_number": cell_space["cell_number"],
        "forward_cell_number": cell_space["forward_boundary_number"],
        "backward_cell_number": cell_space["backward_boundary_number"],
        # Forward leg numbering
        "n_forward_legs": forward_leg_space["n_legs"],
        "forward_leg_interior_number": forward_leg_space["interior_leg_number"],
        "forward_leg_boundary_number": forward_leg_space["boundary_leg_number"],
        "forward_leg_weights": build_leg_weight_vector(forward_leg_space),
        # Forward interpolation operator in CSR format
        "forward_weights": forward_operator["weights"],
        "forward_columns": forward_operator["columns"],
        "forward_rows": forward_operator["rows"],
        # Backward leg numbering
        "n_backward_legs": backward_leg_space["n_legs"],
        "backward_leg_interior_number": backward_leg_space["interior_leg_number"],
        "backward_leg_boundary_number": backward_leg_space["boundary_leg_number"],
        "backward_leg_weights": build_leg_weight_vector(backward_leg_space),
        # Backward interpolation operator in CSR format
        "backward_weights": backward_operator["weights"],
        "backward_columns": backward_operator["columns"],
        "backward_rows": backward_operator["rows"],
    }


__all__ = [
    "assign_cell_space_numbers",
    "build_leg_space_metadata",
    "build_leg_weight_vector",
    "build_leg_destination_operator",
    "calculate_parallel_map_operators",
]
