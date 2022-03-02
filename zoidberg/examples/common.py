import zoidberg as zb
import numpy as np
from boututils.datafile import DataFile
import boututils.calculus as calc


# calculate curvature for curvilinear grids
def calc_curvilinear_curvature(fname, field, grid, maps):
    f = DataFile(str(fname), write=True)
    B = f.read("B")

    dx = grid.metric()["dx"]
    dz = grid.metric()["dz"]
    g_11 = grid.metric()["g_xx"]
    g_22 = grid.metric()["g_yy"]
    g_33 = grid.metric()["g_zz"]
    g_12 = 0.0
    g_13 = grid.metric()["g_xz"]
    g_23 = 0.0

    GR = np.zeros(B.shape)
    GZ = np.zeros(B.shape)
    Gphi = np.zeros(B.shape)
    dRdz = np.zeros(B.shape)
    dZdz = np.zeros(B.shape)
    dRdx = np.zeros(B.shape)
    dZdx = np.zeros(B.shape)

    for y in np.arange(0, B.shape[1]):
        pol, _ = grid.getPoloidalGrid(y)
        R = pol.R
        Z = pol.Z
        # G = \vec{B}/B, here in cylindrical coordinates
        GR[:, y, :] = field.Bxfunc(R, y, Z) / ((B[:, y, :]) ** 2)
        GZ[:, y, :] = field.Bzfunc(R, y, Z) / ((B[:, y, :]) ** 2)
        Gphi[:, y, :] = field.Byfunc(R, y, Z) / ((B[:, y, :]) ** 2)
        for x in np.arange(0, B.shape[0]):
            dRdz[x, y, :] = calc.deriv(R[x, :]) / dz[x, y, :]
            dZdz[x, y, :] = calc.deriv(Z[x, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dRdx[:, y, z] = calc.deriv(R[:, z]) / dx[:, y, z]
            dZdx[:, y, z] = calc.deriv(Z[:, z]) / dx[:, y, z]

    R = f.read("R")
    Z = f.read("Z")
    dy = f.read("dy")

    # calculate Jacobian and contravariant terms in curvilinear coordinates
    J = R * (dZdz * dRdx - dZdx * dRdz)
    Gx = (GR * dZdz - GZ * dRdz) * (R / J)
    Gz = (GZ * dRdx - GR * dZdx) * (R / J)

    G_x = Gx * g_11 + Gphi * g_12 + Gz * g_13
    G_y = Gx * g_12 + Gphi * g_22 + Gz * g_23
    G_z = Gx * g_13 + Gphi * g_23 + Gz * g_33

    dG_zdy = np.zeros(B.shape)
    dG_ydz = np.zeros(B.shape)
    dG_xdz = np.zeros(B.shape)
    dG_zdx = np.zeros(B.shape)
    dG_ydx = np.zeros(B.shape)
    dG_xdy = np.zeros(B.shape)
    for y in np.arange(0, B.shape[1]):
        for x in np.arange(0, B.shape[0]):
            dG_ydz[x, y, :] = calc.deriv(G_y[x, y, :]) / dz[x, y, :]
            dG_xdz[x, y, :] = calc.deriv(G_x[x, y, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dG_ydx[:, y, z] = calc.deriv(G_y[:, y, z]) / dx[:, y, z]
            dG_zdx[:, y, z] = calc.deriv(G_z[:, y, z]) / dx[:, y, z]

    # this should really use the maps...
    for x in np.arange(0, B.shape[0]):
        for z in np.arange(0, B.shape[-1]):
            dG_zdy[x, :, z] = calc.deriv(G_z[x, :, z]) / dy[x, :, z]
            dG_xdy[x, :, z] = calc.deriv(G_x[x, :, z]) / dy[x, :, z]

    bxcvx = (dG_zdy - dG_ydz) / J
    bxcvy = (dG_xdz - dG_zdx) / J
    bxcvz = (dG_ydx - dG_xdy) / J

    f.write("bxcvx", bxcvx)
    f.write("bxcvy", bxcvy)
    f.write("bxcvz", bxcvz)
    f.write("J", J)
    f.close()


def get_lines(
    field, start_r, start_z, yslices, yperiod=2 * np.pi, npoints=150, smoothing=False
):
    rzcoord, ycoords = zb.fieldtracer.trace_poincare(
        field, start_r, start_z, yperiod, y_slices=yslices, revs=npoints
    )

    lines = []
    for i in range(ycoords.shape[0]):
        r = rzcoord[:, i, 0, 0]
        z = rzcoord[:, i, 0, 1]
        line = zb.rzline.line_from_points(r, z)
        # Re-map the points so they're approximately uniform in distance along the surface
        # Note that this results in some motion of the line
        line = line.equallySpaced()
        lines.append(line)

    return lines


# smooth the metric tensor components
def smooth_metric(
    fname, write_to_file=False, return_values=False, smooth_metric=True, order=7
):
    from scipy.signal import savgol_filter

    f = DataFile(str(fname), write=True)
    bxcvx = f.read("bxcvx")
    bxcvz = f.read("bxcvz")
    bxcvy = f.read("bxcvy")
    J = f.read("J")

    bxcvx_smooth = np.zeros(bxcvx.shape)
    bxcvy_smooth = np.zeros(bxcvy.shape)
    bxcvz_smooth = np.zeros(bxcvz.shape)
    J_smooth = np.zeros(J.shape)

    if smooth_metric:
        g13 = f.read("g13")
        g_13 = f.read("g_13")
        g11 = f.read("g11")
        g_11 = f.read("g_11")
        g33 = f.read("g33")
        g_33 = f.read("g_33")

        g13_smooth = np.zeros(g13.shape)
        g_13_smooth = np.zeros(g_13.shape)
        g11_smooth = np.zeros(g11.shape)
        g_11_smooth = np.zeros(g_11.shape)
        g33_smooth = np.zeros(g33.shape)
        g_33_smooth = np.zeros(g_33.shape)

    for y in np.arange(0, bxcvx.shape[1]):
        for x in np.arange(0, bxcvx.shape[0]):
            bxcvx_smooth[x, y, :] = savgol_filter(
                bxcvx[x, y, :], np.int(np.ceil(bxcvx.shape[-1] / 2) // 2 * 2 + 1), order
            )
            bxcvz_smooth[x, y, :] = savgol_filter(
                bxcvz[x, y, :], np.int(np.ceil(bxcvz.shape[-1] / 2) // 2 * 2 + 1), order
            )
            bxcvy_smooth[x, y, :] = savgol_filter(
                bxcvy[x, y, :], np.int(np.ceil(bxcvy.shape[-1] / 2) // 2 * 2 + 1), order
            )
            J_smooth[x, y, :] = savgol_filter(
                J[x, y, :], np.int(np.ceil(J.shape[-1] / 2) // 2 * 2 + 1), order
            )
            if smooth_metric:
                g11_smooth[x, y, :] = savgol_filter(
                    g11[x, y, :], np.int(np.ceil(g11.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_11_smooth[x, y, :] = savgol_filter(
                    g_11[x, y, :],
                    np.int(np.ceil(g_11.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )
                g13_smooth[x, y, :] = savgol_filter(
                    g13[x, y, :], np.int(np.ceil(g13.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_13_smooth[x, y, :] = savgol_filter(
                    g_13[x, y, :],
                    np.int(np.ceil(g_13.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )
                g33_smooth[x, y, :] = savgol_filter(
                    g33[x, y, :], np.int(np.ceil(g33.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_33_smooth[x, y, :] = savgol_filter(
                    g_33[x, y, :],
                    np.int(np.ceil(g_33.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )

    if write_to_file:
        # f.write('bxcvx',bxcvx_smooth)
        # f.write('bxcvy',bxcvy_smooth)
        # f.write('bxcvz',bxcvz_smooth)
        f.write("J", J_smooth)

        if smooth_metric:
            f.write("g11", g11_smooth)
            f.write("g_11", g_11_smooth)
            f.write("g13", g13_smooth)
            f.write("g_13", g_13_smooth)
            f.write("g33", g33_smooth)
            f.write("g_33", g_33_smooth)

    f.close()
    if return_values:
        return bxcvx_smooth, bxcvy_smooth, bxcvz_smooth, bxcvx, bxcvy, bxcvz


def calc_iota(field, start_r, start_z):
    from scipy.signal import argrelextrema

    toroidal_angle = np.linspace(0.0, 400 * np.pi, 10000, endpoint=False)
    result = zb.fieldtracer.FieldTracer.follow_field_lines(
        field, start_r, start_z, toroidal_angle
    )
    peaks = argrelextrema(result[:, 0, 0], np.greater, order=10)[0]
    iota_bar = 2 * np.pi / (toroidal_angle[peaks[1]] - toroidal_angle[peaks[0]])
    # plt.plot(toroidal_angle, result[:,0,0]); plt.show()
    return iota_bar
