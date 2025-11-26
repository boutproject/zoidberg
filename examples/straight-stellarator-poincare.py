#!/usr/bin/env python

# Create grids for a straight stellarator, based on
# curvilinear grids


import zoidberg

#############################################################################
# Define the magnetic field

# Length in y after which the coils return to their starting (R,Z) locations
yperiod = 10.0

magnetic_field = zoidberg.field.StraightStellarator(
    I_coil=0.4, radius=1.0, yperiod=yperiod
)

zoidberg.plot.plot_poincare(magnetic_field, 0.3, 0.0, yperiod)
