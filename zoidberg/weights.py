"""Routines to calculate interpolation weights

N cells divided between Ne evolving cells and Nb = N - Ne boundary cells

    [0 .. evolving cells .. (Ne - 1) | Ne .. boundary cells .. (N - 1)]

Boundary cells include both radial (X) boundaries and parallel (Yup/Ydown) cells.

Cell index numbers are stored in three arrays:

    cell_number[x,y,z]
    cell_number_yup[x,y,z]
    cell_number_ydown[x,y,z]

Non-negative numbers indicate a valid cell number.

For forward and backward maps the Nw weights are stored in CSR format

    weights[Nw]
    column_index[Nw]
    row_index[Ne]     <- Starting index into weights and column_index
                         This will be -1 for boundary points
                         Needs to be considered when getting the weights

For each evolving cell i in 0...(Ne-1) the weight index j is
row_index[i]..(row_index[i+1] - 1)

i.e.

result[i] = sum_{j = row_index[i]}^{row_index[i+1]-1} weight[j] * input[column_index[j]]

The column_index cells go from 0..(N-1), including boundary cells.
Note: row_index[i+1] may be -1, so skip over -1 entries.

Note that these operators are usually represented as non-square
matrices: Input (columns) of length N, output (rows) of length Ne < N.
This is because boundary conditions are set independently.


The output grid file will contain
        int cell_number(x, y, z) ;
        int total_cells ;

        int forward_cell_number(x, y, z) ;
        double forward_weights(t) ;
        int forward_columns(t) ;
        int forward_rows(t2) ;

        int backward_cell_number(x, y, z) ;
        double backward_weights(t3) ;
        int backward_columns(t3) ;
        int backward_rows(t2) ;

"""

import numpy as np


def calc_cell_numbers(maps):
    """Given a field line map dictionary, assign numbers to evolving
    cells and boundary cells"""
    nx, ny, nz = maps["R"].shape
    MXG = maps["MXG"]
    # Number of evolving cells
    N_evolving = (nx - 2 * MXG) * ny * nz

    # Numbering
    cell_number_array = np.full((nx, ny, nz), -1, dtype=int)
    cell_number = 0
    for i in range(MXG, nx - MXG):
        for j in range(ny):
            for k in range(nz):
                cell_number_array[i, j, k] = cell_number
                cell_number += 1

    # Inner radial boundary cells
    for i in range(MXG):
        for j in range(ny):
            for k in range(nz):
                cell_number_array[i, j, k] = cell_number
                cell_number += 1
    # Outer radial boundary cells
    for i in range(nx - MXG, nx):
        for j in range(ny):
            for k in range(nz):
                cell_number_array[i, j, k] = cell_number
                cell_number += 1

    backward_cell_number = np.full((nx, ny, nz), -1, dtype=int)
    forward_cell_number = np.full((nx, ny, nz), -1, dtype=int)

    # Iterate through for forward and backward maps
    forward_xt_prime = maps["forward_xt_prime"]
    backward_xt_prime = maps["backward_xt_prime"]

    # Number of radial boundary cells
    N_radial = 2 * MXG * ny * nz
    cell_number = N_evolving + N_radial

    # Add the parallel boundary cells
    for i in range(MXG, nx - MXG):
        for j in range(ny):
            for k in range(nz):
                if backward_xt_prime[i, j, k] < MXG:
                    backward_cell_number[i, j, k] = cell_number
                    cell_number += 1
                elif backward_xt_prime[i, j, k] > nx - MXG - 1:
                    backward_cell_number[i, j, k] = cell_number
                    cell_number += 1
                if forward_xt_prime[i, j, k] < MXG:
                    forward_cell_number[i, j, k] = cell_number
                    cell_number += 1
                elif forward_xt_prime[i, j, k] > nx - MXG - 1:
                    forward_cell_number[i, j, k] = cell_number
                    cell_number += 1

    return {
        "N_cells": cell_number,
        "N_evolving": N_evolving,
        "cell_number": cell_number_array,
        "forward_cell_number": forward_cell_number,
        "backward_cell_number": backward_cell_number,
    }


def calc_interpolation(numbering, maps):
    """
    Calculate CSR format matrix representing a 2D (X-Z) interpolation operation

    Implements the cubic Catmull-Rom spline
    Coefficients taken from https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    """

    cell_number = numbering["cell_number"]
    MXG = maps["MXG"]

    # Offsets and weights for 1D interpolation
    offsets1D = [-1, 0, 1, 2]

    def weights1D(u):
        return 0.5 * np.array(
            [
                -(u**3) + 2.0 * u**2 - u,
                3.0 * u**3 - 5.0 * u**2 + 2,
                -3.0 * u**3 + 4.0 * u**2 + u,
                u**3 - u**2,
            ]
        )

    # At the end the operators will be stored in CSR
    # format. When assembling the rows are not set in order
    # so a more flexible format is needed.
    # To use: Append (col, weight) pairs to the row lists.
    class Matrix:
        def __init__(self, N):
            self.rows = [[] for _ in range(N)]

        def csr(self):
            """Calculates CSR format arrays"""
            rows = np.full(len(self.rows), -1, dtype=int)
            weights = []
            columns = []
            for i, entries in enumerate(self.rows):
                if len(entries) == 0:
                    continue
                rows[i] = len(weights)
                for col, weight in entries:
                    columns.append(col)
                    weights.append(weight)
            return {"weights": weights, "columns": columns, "rows": rows}

    forward_mat = Matrix(numbering["N_cells"])
    backward_mat = Matrix(numbering["N_cells"])

    nx, ny, nz = cell_number.shape

    # Calculate first forward interpolation coefficients then backward
    #
    # This looks complicated because the rows corresponding to boundary
    # cells in the forward direction are set using the backward mapping.
    #
    # Rows in both forward_mat and backward_mat are set from
    # bndry_cell_number
    def calc(mat, rev_mat, yoffset, xtarr, ztarr, bndry_cell_number):
        for i in range(MXG, nx - MXG):
            for j in range(ny):
                for k in range(nz):
                    row = cell_number[i, j, k]
                    assert row >= 0

                    xt = xtarr[i, j, k]
                    zt = ztarr[i, j, k]
                    if (xt < MXG) or (xt > nx - MXG - 1):
                        # Boundary => Take value from yup/ydown boundary cell
                        bndry_row = bndry_cell_number[i, j, k]
                        assert bndry_row >= 0
                        mat.rows[row] = [(bndry_row, 1.0)]

                        # Boundary cells "free" boundary extrapolate
                        mat.rows[bndry_row] = [(row, -1.0), (bndry_row, 2.0)]

                        # In the reverse matrix e.g. going forward
                        # from the backward boundary cell. This is
                        # included because the divergence operator is
                        # the transpose of the gradient.  If the
                        # divergence in an evolving cell depends on a
                        # boundary cell, then the gradient in the
                        # boundary cell must depend on the evolving
                        # cell.
                        rev_mat.rows[bndry_row] = [(row, 1.0)]
                    else:
                        # Not a boundary point => Interpolating
                        xi = int(xt)  # Floor
                        zi = int(zt)

                        weights_x = weights1D(xt - xi)
                        weights_z = weights1D(zt - zi)

                        # Ensure that points in the X boundary are not used
                        for wi in range(len(offsets1D)):
                            if xi + offsets1D[wi] < MXG:
                                # Move weight away from boundary
                                weights_x[wi + 1] += weights_x[wi]
                                weights_x[wi] = 0.0
                        for wi in reversed(range(len(offsets1D))):
                            if xi + offsets1D[wi] >= nx - MXG:
                                weights_x[wi - 1] += weights_x[wi]
                                weights_x[wi] = 0.0

                        # Weights should sum to 1
                        assert abs(np.sum(weights_x) - 1) < 1e-8
                        assert abs(np.sum(weights_z) - 1) < 1e-8

                        for xo, xw in zip(offsets1D, weights_x):
                            if abs(xw) < 1e-10:
                                continue
                            for zo, zw in zip(offsets1D, weights_z):
                                col = cell_number[
                                    np.clip(xi + xo, 0, nx - 1),
                                    (j + yoffset + ny) % ny,
                                    (zi + zo + nz) % nz,
                                ]
                                assert col >= 0

                                mat.rows[row].append((col, xw * zw))

    calc(
        forward_mat,
        backward_mat,
        +1,
        maps["forward_xt_prime"],
        maps["forward_zt_prime"],
        numbering["forward_cell_number"],
    )

    calc(
        backward_mat,
        forward_mat,
        -1,
        maps["backward_xt_prime"],
        maps["backward_zt_prime"],
        numbering["backward_cell_number"],
    )

    return forward_mat.csr(), backward_mat.csr()


def calc_weights(maps):
    """
    Calculate interpolation weights for forward and backward maps.

    Returns a dictionary of arrays to be read into BOUT++
    """

    numbering = calc_cell_numbers(maps)

    forward, backward = calc_interpolation(numbering, maps)

    return {
        "cell_number": numbering["cell_number"],
        "total_cells": numbering["N_cells"],
        "forward_cell_number": numbering["forward_cell_number"],
        "forward_weights": forward["weights"],
        "forward_columns": forward["columns"],
        "forward_rows": forward["rows"],
        "backward_cell_number": numbering["backward_cell_number"],
        "backward_weights": backward["weights"],
        "backward_columns": backward["columns"],
        "backward_rows": backward["rows"],
    }
