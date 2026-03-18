from itertools import chain, product

import numpy as np

from . import field, grid, zoidberg


def test_make_maps_slab():
    nx = 5
    ny = 8
    nz = 7

    # Create a straight magnetic field in a slab
    straight_field = field.Slab(By=1.0, Bz=0.0, Bzprime=0.0)

    # Create a rectangular grid in (x,y,z)
    rectangle = grid.rectangular_grid(nx, ny, nz)

    # Two parallel slices in each direction
    nslice = 2

    # Calculate forwards and backwards maps
    maps = zoidberg.make_maps(rectangle, straight_field, nslice=nslice)

    # Since this is a straight magnetic field in a simple rectangle,
    # all the maps should be the same, and should be the identity
    identity_map_x, identity_map_z = np.meshgrid(
        np.arange(nx), np.arange(nz), indexing="ij"
    )

    # Check that maps has the required forward and backward index variables
    offsets = chain(range(1, nslice + 1), range(-1, -(nslice + 1), -1))
    field_line_maps = ["xt_prime", "zt_prime"]

    for field_line_map, offset in product(field_line_maps, offsets):
        var = zoidberg.parallel_slice_field_name(field_line_map, offset)
        print("Current field: ", var)
        assert var in maps

        # Each map should have the same shape as the grid
        assert maps[var].shape == (nx, ny, nz)

        # The first/last abs(offset) points are not valid, so ignore those
        interior_range = (
            range(ny - abs(offset)) if offset > 0 else range(abs(offset), ny)
        )
        # Those invalid points should be set to -1
        end_slice = slice(-1, -(offset + 1), -1) if offset > 0 else slice(0, -offset)
        identity_map = identity_map_x if "x" in var else identity_map_z

        for y in interior_range:
            assert np.allclose(maps[var][:, y, :], identity_map)

        # The end slice should hit a boundary
        assert np.allclose(maps[var][:, end_slice, :], -1.0)


def test_make_maps_straight_stellarator():
    nx = 5
    ny = 6
    nz = 7

    # Create magnetic field
    magnetic_field = field.StraightStellarator(radius=np.sqrt(2.0))

    # Create a rectangular grid in (x,y,z)
    rectangle = grid.rectangular_grid(
        nx, ny, nz, Lx=1.0, Lz=1.0, Ly=10.0, yperiodic=True
    )

    # Here both the field and and grid are centred at (x,z) = (0,0)
    # and the rectangular grid here fits entirely within the coils

    zoidberg.make_maps(rectangle, magnetic_field)


def test_repair_sparse_boundary_holes_repairs_narrow_adjacent_slice():
    nx = 5
    ny = 3
    nz = 8
    xt_base = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz)).astype(float)
    z_base = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz)).astype(float)
    maps = {
        "forward_xt_prime": xt_base.copy(),
        "forward_zt_prime": z_base.copy(),
        "forward_R": xt_base.copy(),
        "forward_Z": z_base.copy(),
        "backward_xt_prime": xt_base.copy(),
        "backward_zt_prime": z_base.copy(),
        "backward_R": xt_base.copy(),
        "backward_Z": z_base.copy(),
    }

    maps["forward_xt_prime"][0, :, :] = -1.0
    maps["backward_xt_prime"][0, :, :] = -1.0
    maps["forward_zt_prime"][0, :, :] = -1.0
    maps["backward_zt_prime"][0, :, :] = -1.0
    maps["forward_xt_prime"][1, 1, [2, 3]] = -1.0
    maps["backward_xt_prime"][1, 1, [4]] = -1.0

    zoidberg._repair_sparse_boundary_holes(maps, nx, nz)

    assert np.all(maps["forward_xt_prime"][1] >= 0.0)
    assert np.all(maps["backward_xt_prime"][1] >= 0.0)
    assert np.allclose(maps["forward_xt_prime"][1, 1, 2:4], 1.0)
    assert np.allclose(maps["backward_xt_prime"][1, 1, 4], 1.0)
    assert np.all(maps["forward_xt_prime"][0] == -1.0)


def test_repair_sparse_boundary_holes_preserves_broad_invalid_region():
    nx = 5
    ny = 2
    nz = 10
    xt_base = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz)).astype(float)
    z_base = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz)).astype(float)
    maps = {
        "forward_xt_prime": xt_base.copy(),
        "forward_zt_prime": z_base.copy(),
        "forward_R": xt_base.copy(),
        "forward_Z": z_base.copy(),
        "backward_xt_prime": xt_base.copy(),
        "backward_zt_prime": z_base.copy(),
        "backward_R": xt_base.copy(),
        "backward_Z": z_base.copy(),
    }

    maps["forward_xt_prime"][0, :, :] = -1.0
    maps["forward_xt_prime"][1, :, :5] = -1.0

    before = maps["forward_xt_prime"].copy()
    zoidberg._repair_sparse_boundary_holes(maps, nx, nz)

    assert np.array_equal(maps["forward_xt_prime"], before)
