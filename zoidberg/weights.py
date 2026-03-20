"""Construct interpolation operators for field-line maps.

This module assigns global cell numbers and builds forward and backward
interpolation operators for BOUT++ mesh files.

Numbering scheme
----------------
Cells are numbered in three arrays:

- ``cell_number[x, y, z]``:
    Numbering for evolving cells and radial X-boundary cells.
- ``forward_boundary_number[x, y, z]``:
    Numbering for forward parallel boundary cells (``yup``).
- ``backward_boundary_number[x, y, z]``:
    Numbering for backward parallel boundary cells (``ydown``).

A value of ``-1`` means that no numbered cell exists at that location.

The numbering order is:

1. evolving cells in the interior ``x = MXG .. nx-MXG-1``
2. inner radial boundary cells
3. outer radial boundary cells
4. backward parallel boundary cells
5. forward parallel boundary cells

Operator storage format
-----------------------
The forward and backward operators are stored as sparse matrices over
all numbered cells and exported in **standard CSR (Compressed Sparse Row)**
format with three arrays:

- ``weights[nnz]``:
    nonzero matrix entries
- ``columns[nnz]``:
    column indices corresponding to ``weights``
- ``rows[n_rows + 1]``:
    CSR row pointer array

Row ``i`` occupies entries

``weights[rows[i] : rows[i+1]]``

with column indices

``columns[rows[i] : rows[i+1]]``.

Empty rows satisfy ``rows[i] == rows[i+1]``.

Interpolation
-------------
Interior interpolation in X-Z uses a 4x4 Catmull-Rom stencil. The Y direction
is shifted by ``+1`` for the forward map and ``-1`` for the backward map.

If a mapped point lands in the radial boundary region, the operator uses the
corresponding parallel boundary cell instead of an interior interpolation
stencil.
"""

import numpy as np


def any_hits_radial_boundary(x_mapped, nx, mxg):
    """Return True if a mapped X coordinate lies in the radial boundary region."""
    return np.any(x_mapped < mxg) or np.any(x_mapped > nx - mxg - 1)


def assign_cell_numbers(maps):
    """Assign global numbers to interior and boundary cells.

    Parameters
    ----------
    maps : dict
        Field-line map dictionary. Required entries are:
        ``R``, ``MXG``, ``forward_xt_prime``, and ``backward_xt_prime``.

    Returns
    -------
    dict
        Dictionary containing:
        - ``N_cells``: total number of numbered cells
        - ``N_evolving``: number of evolving interior cells
        - ``cell_number``: numbering for interior and radial boundary cells
        - ``forward_boundary_number``: numbering for forward parallel boundaries
        - ``backward_boundary_number``: numbering for backward parallel boundaries

    Notes
    -----
    Radial X-boundary cells are numbered directly in ``cell_number``.
    Parallel boundary cells are numbered separately in
    ``forward_boundary_number`` and ``backward_boundary_number``.
    """
    nx, ny, nz = maps["R"].shape
    mxg = maps["MXG"]

    n_evolving = (nx - 2 * mxg) * ny * nz

    cell_number = np.full((nx, ny, nz), -1, dtype=int)

    next_cell = 0

    # Interior evolving cells
    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    # Inner radial boundary cells
    for x in range(mxg):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    # Outer radial boundary cells
    for x in range(nx - mxg, nx):
        for y in range(ny):
            for z in range(nz):
                cell_number[x, y, z] = next_cell
                next_cell += 1

    backward_boundary_number = np.full((nx, ny, nz), -1, dtype=int)
    forward_boundary_number = np.full((nx, ny, nz), -1, dtype=int)

    # Arrays of mapped indices [x,y,z,s]
    # where 's' is the sub-cell node index
    forward_xts = maps["forward_xt_primes"]
    backward_xts = maps["backward_xt_primes"]

    # Cell center maps that are used to set boundary conditions
    forward_xt = maps["forward_xt_prime"]
    backward_xt = maps["backward_xt_prime"]

    # Parallel boundary numbering starts after all interior + radial cells
    for x in range(mxg, nx - mxg):
        for y in range(ny):
            for z in range(nz):
                # Allocate a boundary cell if any sub-cell points hit the boundary
                if any_hits_radial_boundary(backward_xts[x, y, z, :], nx, mxg):
                    backward_boundary_number[x, y, z] = next_cell
                    next_cell += 1
                    # Ensure that the cell is marked as a boundary
                    if not any_hits_radial_boundary(backward_xt[x, y, z], nx, mxg):
                        # Find a point that hits the boundary
                        for s in range(backward_xts.shape[-1]):
                            if any_hits_radial_boundary(
                                backward_xts[x, y, z, s], nx, mxg
                            ):
                                backward_xt[x, y, z] = backward_xts[x, y, z, s]
                                break

                if any_hits_radial_boundary(forward_xts[x, y, z, :], nx, mxg):
                    forward_boundary_number[x, y, z] = next_cell
                    next_cell += 1
                    if not any_hits_radial_boundary(forward_xt[x, y, z], nx, mxg):
                        # Find a point that hits the boundary
                        for s in range(forward_xts.shape[-1]):
                            if any_hits_radial_boundary(
                                forward_xts[x, y, z, s], nx, mxg
                            ):
                                forward_xt[x, y, z] = forward_xts[x, y, z, s]
                                break

    return {
        "N_cells": next_cell,
        "N_evolving": n_evolving,
        "cell_number": cell_number,
        "forward_boundary_number": forward_boundary_number,
        "backward_boundary_number": backward_boundary_number,
    }


def catmull_rom_weights(u):
    """Return 1D Catmull-Rom interpolation weights for fractional offset ``u``.

    Parameters
    ----------
    u : float
        Fractional coordinate relative to the left stencil point. Typically
        in the interval ``[0, 1)``.

    Returns
    -------
    numpy.ndarray
        Four Catmull-Rom weights corresponding to offsets ``[-1, 0, 1, 2]``.
    """
    return 0.5 * np.array(
        [
            -(u**3) + 2.0 * u**2 - u,
            3.0 * u**3 - 5.0 * u**2 + 2.0,
            -3.0 * u**3 + 4.0 * u**2 + u,
            u**3 - u**2,
        ]
    )


def shift_weights_out_of_x_boundaries(weights_x, x_base, x_offsets, nx, mxg):
    """Move X interpolation weight off radial boundary points.

    Parameters
    ----------
    weights_x : numpy.ndarray
        1D interpolation weights for X offsets.
    x_base : int
        Floor of the mapped X coordinate.
    x_offsets : sequence[int]
        Stencil offsets corresponding to ``weights_x``.
    nx : int
        Number of X points.
    mxg : int
        Width of the radial boundary region.

    Returns
    -------
    numpy.ndarray
        Adjusted X weights that do not reference radial boundary cells.

    Notes
    -----
    If a stencil point falls into the inner radial boundary, its weight is
    added to the nearest point to the right. If it falls into the outer
    radial boundary, its weight is added to the nearest point to the left.
    """
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
    """Accumulate sparse rows before exporting to CSR format.

    Parameters
    ----------
    n_rows : int
        Number of matrix rows.

    Notes
    -----
    Rows are stored internally as Python lists of ``(column, weight)`` pairs.
    This allows rows to be assembled out of order during operator construction.

    The exported format is **standard CSR (Compressed Sparse Row)**:

    * ``weights[nnz]`` — nonzero matrix entries
    * ``columns[nnz]`` — column indices corresponding to ``weights``
    * ``rows[n_rows + 1]`` — CSR row pointer array

    The CSR invariant is:

    ``rows[i]`` gives the starting index of row ``i`` in ``weights`` and
    ``columns``, and ``rows[i+1]`` gives the end index.

    Therefore

    ``rows[i+1] - rows[i]``

    is the number of nonzero entries in row ``i``.

    Empty rows are represented by

    ``rows[i] == rows[i+1]``.
    """

    def __init__(self, n_rows):
        self.rows = [[] for _ in range(n_rows)]

    def set_row(self, row, entries):
        """Assign a row exactly once.

        Parameters
        ----------
        row : int
            Row index to assign.
        entries : list[tuple[int, float]]
            Sparse row entries.

        Raises
        ------
        ValueError
            If the row has already been assigned.
        """
        if self.rows[row]:
            raise ValueError(f"Row {row} assigned more than once")

        self.rows[row] = list(entries)

    def add_to_entry(self, row, col, weight):
        """Add to an entry. If there is already an entry
        at this (row, col) then it will be added."""
        for i, (entry_col, entry_weight) in enumerate(self.rows[row]):
            if entry_col == col:
                # Replace this entry with the sum
                self.rows[row][i] = (col, entry_weight + weight)
                return
        self.rows[row].append((col, weight))

    def to_csr(self):
        """Export matrix to standard CSR format using preallocated arrays.

        Returns
        -------
        dict
            Dictionary containing

            * ``weights`` : numpy.ndarray
            Nonzero matrix entries.
            * ``columns`` : numpy.ndarray
            Column indices corresponding to ``weights``.
            * ``rows`` : numpy.ndarray
            CSR row pointer array of length ``n_rows + 1``.

        Notes
        -----
        This implementation performs two passes:

        1. Count nonzeros in each row to construct the CSR row pointer.
        2. Allocate arrays of the correct size and fill them.

        This avoids Python list growth during export and is significantly
        faster for large matrices.
        """

        n_rows = len(self.rows)

        # ---- pass 1: count nonzeros ----

        nnz_per_row = np.fromiter(
            (len(r) for r in self.rows),
            dtype=int,
            count=n_rows,
        )

        row_ptr = np.empty(n_rows + 1, dtype=int)
        row_ptr[0] = 0
        np.cumsum(nnz_per_row, out=row_ptr[1:])

        nnz = row_ptr[-1]

        # ---- allocate arrays ----

        weights = np.empty(nnz, dtype=float)
        columns = np.empty(nnz, dtype=int)

        # ---- pass 2: fill arrays ----

        for row, entries in enumerate(self.rows):
            start = row_ptr[row]

            for i, (col, weight) in enumerate(entries):
                columns[start + i] = col
                weights[start + i] = weight

        return {
            "weights": weights,
            "columns": columns,
            "rows": row_ptr,
        }


def add_boundary_or_interpolation_row(
    matrix_builder: SparseRowMatrixBuilder,
    reverse_builder: SparseRowMatrixBuilder,
    row: int,
    mapped_xs,
    mapped_zs,
    weights,
    y: int,
    y_offset: int,
    boundary_row: int,
    cell_number,
    mxg: int,
):
    """Add one operator row for a boundary map, interior interpolation,
    or weighted combination.

    Parameters
    ----------
    matrix_builder : SparseRowMatrixBuilder
        Matrix currently being assembled.
    reverse_builder : SparseRowMatrixBuilder
        Opposite-direction matrix, used to set the reverse boundary row.
    row : int
        Output row index for the evolving cell.
    mapped_xs, mapped_xs : [float]
        Mapped X and Z coordinates for all sub-cell node indices.
    weights : [float]
        Weighting factor for each sub-cell node. Sums to 1.
    y : int
        Source Y index.
    y_offset : int
        Y shift to apply: ``+1`` for forward, ``-1`` for backward.
    boundary_row : int
        Parallel boundary cell number for this mapped point, or ``-1`` if none.
    cell_number : numpy.ndarray
        Cell-number array for interior and radial boundary cells.
    mxg : int
        Radial boundary width.

    Returns
    -------
    boundary_weight: float
        A factor between 0 and 1 that should be applied to the volume
        of boundary cells when constructing divergence operators.
        It is the fraction of the cell that hits the boundary.

    Notes
    -----
    If one or more mapped points land in the radial boundary region,
    the row is set to reference the corresponding parallel boundary cell.

    The reverse operator also receives a boundary row contribution. This row is
    expected to be unique; if the same reverse boundary row is assigned twice,
    an exception is raised.
    """
    assert len(weights) == len(mapped_xs)
    assert len(weights) == len(mapped_zs)
    if abs(np.sum(weights) - 1.0) >= 1e-8:
        raise ValueError(f"weights should sum to 1, got {np.sum(weights)}")

    nx, ny, nz = cell_number.shape

    x_offsets = [-1, 0, 1, 2]

    # Track whether any sub-cell nodes hit boundaries
    any_hit_boundary = False
    # Sum of the weights of points hitting the boundary
    boundary_weight = 0.0

    # Iterate over sub-cell nodes
    for mapped_x, mapped_z, subcell_weight in zip(mapped_xs, mapped_zs, weights):
        if any_hits_radial_boundary(mapped_x, nx, mxg):
            any_hit_boundary = True
            boundary_weight += subcell_weight
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
                        f"Invalid column index at "
                        f"x={x_base + x_offset}, y={(y + y_offset + ny) % ny}, z={(z_base + z_offset + nz) % nz}"
                    )

                # Sub-cell nodes will provide entries at overlapping
                # (row, col) so add to existing entries.
                matrix_builder.add_to_entry(
                    row, col, subcell_weight * x_weight * z_weight
                )

    if any_hit_boundary:
        # One or more points hit a boundary
        if boundary_row < 0:
            raise ValueError(
                f"Expected a valid boundary row for mapped_x={mapped_xs}, got {boundary_row}"
            )

        matrix_builder.add_to_entry(row, boundary_row, boundary_weight)
        # Note: Identity operation in the boundary
        #       Extrapolating messes up the boundary fluxes!
        matrix_builder.set_row(
            boundary_row,
            [
                (boundary_row, 1.0),
            ],
        )
        reverse_builder.set_row(boundary_row, [(row, 1.0)])
    return boundary_weight


def build_parallel_interpolation_operators(numbering, maps):
    """Build forward and backward interpolation operators.

    Parameters
    ----------
    numbering : dict
        Output of :func:`assign_cell_numbers`.
    maps : dict
        Field-line map dictionary.

    Returns
    -------
    tuple[dict, dict]
        Two dictionaries in row-start sparse format,
        and 3D arrays of volume fractions for forward
        and backward boundaries.
        ``(forward_operator, backward_operator,
           forward_fraction, backward_fraction)``.

    Notes
    -----
    The forward operator shifts in Y by ``+1`` using
    ``forward_xt_prime`` / ``forward_zt_prime``.
    The backward operator shifts in Y by ``-1`` using
    ``backward_xt_prime`` / ``backward_zt_prime``.
    """
    cell_number = numbering["cell_number"]
    nx, ny, nz = cell_number.shape
    mxg = maps["MXG"]

    forward_builder = SparseRowMatrixBuilder(numbering["N_cells"])
    backward_builder = SparseRowMatrixBuilder(numbering["N_cells"])

    forward_boundary_fraction = np.zeros(cell_number.shape)
    backward_boundary_fraction = np.zeros(cell_number.shape)

    def assemble_direction(
        matrix_builder,
        reverse_builder,
        y_offset,
        xt_primes,
        zt_primes,
        weights,
        boundary_number,
        boundary_fraction,
    ):
        """Assemble one directional interpolation operator."""
        for x in range(mxg, nx - mxg):
            for y in range(ny):
                for z in range(nz):
                    row = cell_number[x, y, z]
                    if row < 0:
                        raise ValueError(f"Invalid evolving row at x={x}, y={y}, z={z}")

                    # Store the fraction [0,1] of the cell that hits
                    boundary_fraction[x, y, z] = add_boundary_or_interpolation_row(
                        matrix_builder=matrix_builder,
                        reverse_builder=reverse_builder,
                        row=row,
                        mapped_xs=xt_primes[x, y, z, :],
                        mapped_zs=zt_primes[x, y, z, :],
                        weights=weights,
                        y=y,
                        y_offset=y_offset,
                        boundary_row=boundary_number[x, y, z],
                        cell_number=cell_number,
                        mxg=mxg,
                    )

    assemble_direction(
        forward_builder,
        backward_builder,
        +1,
        maps["forward_xt_primes"],
        maps["forward_zt_primes"],
        maps["subcell_weights"],
        numbering["forward_boundary_number"],
        forward_boundary_fraction,
    )

    assemble_direction(
        backward_builder,
        forward_builder,
        -1,
        maps["backward_xt_primes"],
        maps["backward_zt_primes"],
        maps["subcell_weights"],
        numbering["backward_boundary_number"],
        backward_boundary_fraction,
    )

    return (
        forward_builder.to_csr(),
        backward_builder.to_csr(),
        forward_boundary_fraction,
        backward_boundary_fraction,
    )


def calculate_parallel_map_operators(maps):
    """Calculate numbering arrays and interpolation operators for BOUT++
    PetscOperators.

    Parameters
    ----------
    maps : dict
        Field-line map dictionary. Uses `forward_{x,z}t_primes` and
        `backward_{x,z}t_primes` if present, falling back to
        `forward_{x,z}t_prime` and `backward_{x,z}t_prime`.

    Returns
    -------
    dict
        Dictionary containing numbering arrays and sparse operator arrays
        ready to be written to a BOUT++ mesh file.
    """
    numbering = assign_cell_numbers(maps)
    forward_op, backward_op, forward_fraction, backward_fraction = (
        build_parallel_interpolation_operators(numbering, maps)
    )

    return {
        "cell_number": numbering["cell_number"],
        "total_cells": numbering["N_cells"],
        "forward_cell_number": numbering["forward_boundary_number"],
        "forward_boundary_fraction": forward_fraction,
        "forward_weights": forward_op["weights"],
        "forward_columns": forward_op["columns"],
        "forward_rows": forward_op["rows"],
        "backward_cell_number": numbering["backward_boundary_number"],
        "backward_boundary_fraction": backward_fraction,
        "backward_weights": backward_op["weights"],
        "backward_columns": backward_op["columns"],
        "backward_rows": backward_op["rows"],
    }
