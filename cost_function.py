import numpy as np
from scipy import interpolate
from interpolation import interpn_linear


########################################################################################################################
#
# Cost function and related functions
#
########################################################################################################################


# Smoothing function
def smooth(v,y,x,i):
    # This function returns the smoothed signal.
    # It takes as inputs:
    # - the input signal v (2D field) to be smoothed
    # - the corresponding spatial coordinates x and y
    # - the step number i which defines the level of smoothing

    v1 = np.zeros(v.shape)
    if len(v.shape) == 2:
        nx, ny = v.shape
    elif len(v.shape) == 3:
        nt, nx, ny = v.shape
    alpha = 0.05 / (2 ** (i*2) + 1)
    for j in np.arange(0, ny):
        j = int(j)
        tloc = y[j]
        kernel_t = np.exp(-((y - tloc) / ny) ** 2 / alpha) / sum(np.exp(-((y - y[int((ny - 1) / 2)]) / ny) ** 2 / alpha))
        for k in np.arange(0, nx):
            k = int(k)
            xloc = x[k]
            kernel_x = np.exp(-((x - xloc) / nx) ** 2 / alpha) / sum(np.exp(-((x - x[int((nx - 1) / 2)]) / nx) ** 2 / alpha))
            if len(v.shape) == 2:
                v1[k,j] = np.dot(np.dot(v,kernel_t),kernel_x)
            elif len(v.shape) == 3:
                for kt in range(v.shape[0]):
                    v1[kt,k,j] = np.dot(np.dot(v[kt,:],kernel_t),kernel_x)
    return v1


#-----------------------------------------------------------------------------------
# Warping functions
def mapped(u,y,x,yyT,xxT,i):
    # This function returns the warped signal on the pixel grid.
    # It takes as inputs:
    # - the input signal u (2D filed)
    # - the corresponding spatial coordinates x and y
    # - the two composant of the mapping yyT and xxT
    # - the corresponding step number i (which is linked to the resolution of the mapping)

    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    if len(yyT.shape) == 2:
        # Transform coordinate
        Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
        Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
        # Interpolated function
        if len(u.shape) == 2:
            uT = interpolate.interpn((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None).reshape(nx,ny)  # ,fill_value=1000)
        elif len(u.shape) == 3:
            uT = np.zeros(u.shape)
            for kt in range(u.shape[0]):
                uT[kt,:,:] = interpolate.interpn((x, y), u[kt,:,:], np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None).reshape(nx,ny)

    elif len(yyT.shape) == 3:
        uT = np.zeros(u.shape)
        for kt in range(u.shape[0]):
            # Transform coordinate
            Tt = interpolate.interpn((x[xc], y[yc]), yyT[:,:,kt], np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
            Tx = interpolate.interpn((x[xc], y[yc]), xxT[:,:,kt], np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
            # Interpolated function
            uT[kt, :, :] = interpolate.interpn((x, y), u[kt, :, :], np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None).reshape(nx, ny)

    return uT


def mapped_TAHMO(u,y,x,yyT,xxT,lat_sta,lon_sta,i):
    # This function returns the warped signalat given coordinates (lat_sta,lon_sta).
    # It takes as inputs:
    # - the input signal u (2D filed)
    # - the corresponding spatial coordinates x and y
    # - the two composant of the mapping yyT and xxT
    # - the coordinates at which to compute the warped signal lat_sta and lon_sta (array)
    # - the step number i (which is linked to the resolution of the mapping)

    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)

    if len(yyT.shape) == 2:
        # Transform coordinate
        Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False ,fill_value=None)
        Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False, fill_value=None)

        # Interpolated function
        if len(u.shape) == 2:
            uT = interpolate.interpn((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None).reshape(nx,ny)  # ,fill_value=1000)
        elif len(u.shape) == 3:
            uT = np.zeros((u.shape[0],len(lat_sta)))
            for kt in range(u.shape[0]):
                uT[kt,:] = interpolate.interpn((x, y), u[kt,:,:], np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None)

    elif len(yyT.shape) == 3:
        uT = np.zeros((u.shape[0],len(lat_sta)))
        for kt in range(u.shape[0]):
            # Transform coordinate
            Tt = interpolate.interpn((x[xc], y[yc]), yyT[:,:,kt], np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False, fill_value=None)
            Tx = interpolate.interpn((x[xc], y[yc]), xxT[:,:,kt], np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False, fill_value=None)
            # Interpolated function
            uT[kt, :] = interpolate.interpn((x, y), u[kt, :, :], np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None)

    return uT


def mapped_weight(u,y,x,yyT,xxT,i):
    # This function returns the warped signal and the corresponding interpolation weight (used in the computation of the derivative of the cost function).
    # It takes as inputs:
    # - the input signal u (time series)
    # - the corresponding time coordinate t
    # - the mapping tT
    # - the corresponding step number i (which is linked to the resolution of the mapping)

    ny = len(y)
    nx = len(x)
    nt = u.shape[0]
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, tt = np.meshgrid(x, y, indexing='ij')

    if len(yyT.shape) == 2:
        # Transform coordinate
        Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')
        Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')
        # Interpolated function
        if len(u.shape) == 2:
            uT, uT_x, uT_y = interpn_linear((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=0)
        elif len(u.shape) == 3:
            uT = np.zeros((nt,nx*ny))
            uT_x = np.zeros((nt,nx*ny))
            uT_y = np.zeros((nt,nx*ny))
            for kt in range(nt):
                uT[kt, :], uT_x[kt, :], uT_y[kt, :] = interpn_linear((x, y), u[kt, :, :], np.array([Tx, Tt]).T, method='linear',
                                                   bounds_error=False, fill_value=None)
            uT = uT.reshape(nt,nx,ny)
            uT_x = uT_x.reshape(-1)
            uT_y = uT_y.reshape(-1)

    elif len(yyT.shape) == 3:
        uT = np.zeros((nt, nx * ny))
        uT_x = np.zeros((nt, nx * ny))
        uT_y = np.zeros((nt, nx * ny))
        for kt in range(nt):
            # Transform coordinate
            Tt = interpolate.interpn((x[xc], y[yc]), yyT[:,:,kt], np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')
            Tx = interpolate.interpn((x[xc], y[yc]), xxT[:,:,kt], np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')
            # Interpolated function
            A,B,C = interpn_linear((x, y), u[kt, :, :], np.array([Tx, Tt]).T,method='linear',
                                                                 bounds_error=False, fill_value=None)
            uT[kt, :], uT_x[kt, :], uT_y[kt, :] = interpn_linear((x, y), u[kt, :, :], np.array([Tx, Tt]).T,method='linear',
                                                                 bounds_error=False, fill_value=None)
        uT = uT.reshape(nt, nx, ny)
        uT_x = uT_x.reshape(-1)
        uT_y = uT_y.reshape(-1)

    return uT, uT_x, uT_y


#-----------------------------------------------------------------------------------
# Derivative function (used in cost_function)

def dXdT(t, x, i):
    nt = len(t)
    nx = len(x)
    mi = 2**i+1
    dnx = int((nx - 1) / 2 ** i)
    dnt = int((nt - 1) / 2 ** i)
    x1 = x[:, np.newaxis]

    dxdT2 = np.zeros((nx * nt, 2 * mi * mi))
    dtdT2 = np.zeros((nx * nt, 2 * mi * mi))
    for j in range(1, mi - 1):
        for k in range(1, mi - 1):
            B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 > x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t > t[dnt * j])
            B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t > t[dnt * j])
            B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 > x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
            B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
            dxdT2[:, k * mi + j] = (B1 + B2 + B3 + B4).reshape(-1)

        k = 0
        B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t < t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, j] = (B1 + B3).reshape(-1)
        k = mi - 1
        B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t < t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, (mi - 1) * mi + j] = (B2 + B4).reshape(-1)

    j = 0
    for k in range(1, mi - 1):
        B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 < x1[dnx * k]) * ( x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        dxdT2[:, k * mi] = (B1 + B2).reshape(-1)
    j = mi - 1
    for k in range(1, mi - 1):
        B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
        B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ( (x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 < x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, k * mi + j] = (B3 + B4).reshape(-1)

    j = 0
    k = 0
    B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * ( t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
    dxdT2[:, k * mi + j] = (B1).reshape(-1)
    k = mi - 1
    B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
    dxdT2[:, k * mi + j] = (B2).reshape(-1)
    k = 0
    j = mi - 1
    B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
    dxdT2[:, k * mi + j] = (B3).reshape(-1)
    k = mi - 1
    B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
    dxdT2[:, k * mi + j] = (B4).reshape(-1)

    dtdT2[:, mi * mi:2 * mi * mi] = dxdT2[:, 0:mi * mi]

    return dxdT2, dtdT2


#-----------------------------------------------------------------------------------
# Constraint functions

def constr1(grid,i):
    # Check if the first constraint is respected
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2**i+1,2**i+1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2**i+1,2**i+1))
    c1 = (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 0:mi - 1]) * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]) - (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,0:mi - 1]) * (xxT[1:mi,1:mi] - xxT[0:mi - 1,0:mi - 1])
    return c1.reshape(-1)

def constr2(grid,i):
    # Check if the second constraint is respected
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c2 = (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]) * (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]) - (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]) * (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1])
    return c2.reshape(-1)

def constr3(grid,i):
    # Check if the third constraint is respected
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c3 = (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]) * (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]) - (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]) * (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi])
    return c3.reshape(-1)

def constr4(grid,i):
    # Check if the foutth constraint is respected
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c4 = (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]) * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]) - (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]) * (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi])
    return c4.reshape(-1)


#-----------------------------------------------------------------------------------
# Derivative of the constraint functions

def dc1dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc1 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[1:mi, 0:mi-1] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc1dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)

    dc1dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (xxT[1:mi, 0:mi - 1] - xxT[0:mi-1, 0:mi-1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 0:mi - 1] - xxT[0:mi-1, 0:mi-1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[1:mi,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] +=  (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)

    Jc1[:,0:mi**2] = dc1dx
    Jc1[:,mi**2:] = dc1dt
    return Jc1

def dc2dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc2 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc2dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)

    dc2dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] +=  (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)

    Jc2[:,0:mi**2] = dc2dx
    Jc2[:,mi**2:] = dc2dt
    return Jc2

def dc3dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc3 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[1:mi, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc3dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)

    dc3dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] +=  (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]).reshape(-1)

    Jc3[:,0:mi**2] = dc3dx
    Jc3[:,mi**2:] = dc3dt
    return Jc3

def dc4dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc4 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[1:mi, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))

    dc4dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)

    dc4dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] +=  (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi]).reshape(-1)

    Jc4[:,0:mi**2] = dc4dx
    Jc4[:,mi**2:] = dc4dt
    return Jc4


#-----------------------------------------------------------------------------------
# Cost function

def Jp(grid,b,u,v,y,x,i,c1,c2,c3,dxdT,dydT,Ax,Ay,mask=None,nt0=None):
    # This function returns the value and the derivative of the cost function for the approaches A1 and A2.
    # It takes as input:
    # - the two components of the mapping stored in gridr.
    # - the smoothed inputs us and vs.
    # - the spatial coordinates x and y.
    # - the step i (defining the smoothing and the resolution of the mapping).
    # - the regulation coefficients c1, c2, c3 and ct.
    # - the mask. It needs to have the same dimension as the fields u and v. It is used to mask the area with no data (in case of irregularly spaced observations.
    # - For approach A2: nt0 is the number of time steps with rainfall.

    mi = 2**i + 1

    ny = len(y)
    nx = len(x)
    if len(u.shape) == 2:
        nt = 1
        nt0 = 1
    elif len(u.shape) == 3:
        nt = u.shape[0]
        if nt0 == None:
            nt0 = nt
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')
    yyT = grid[(mi)**2:(mi)**2*2].reshape((mi,mi))
    xxT = grid[0:(mi)**2].reshape((mi, mi))
    b1 = b[0 : (mi-1)**2]
    b2 = b[(mi-1)**2 : 2*(mi - 1)**2]
    b3 = b[2*(mi-1)**2 : 3*(mi - 1)**2]
    b4 = b[3*(mi-1)**2 :  4*(mi - 1)**2]

    v1 = v
    u1 = mapped(u, y, x, yyT, xxT, i)

    ydif = (yyT-yyc).reshape(-1)
    xdif = (xxT-xxc).reshape(-1)

    dT = np.zeros(4 * mi ** 2)
    dT[0:mi ** 2] = Ax @ xdif
    dT[mi ** 2:2 * mi ** 2] = Ay @ xdif
    dT[2 * mi ** 2:3 * mi ** 2] = Ax @ ydif
    dT[3 * mi ** 2:4 * mi ** 2] = Ay @ ydif

    divT = Ax @ xdif + Ay @ ydif

    # Cost
    if mask is None:
        err = (v1 - u1).reshape(-1)
        Jo = np.sqrt(np.sum((v1 - u1) ** 2) ) / np.sqrt(nt0)
    else:
        err = (mask*(v1 - u1)).reshape(-1)
        Jo = np.sqrt(np.sum(mask * (v1 - u1) ** 2)) / np.sqrt(nt0)



    Jb = c1 /(mi) * np.sqrt(np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) \
               + c2 /(mi) * np.sqrt(np.dot(dT.T, dT)) \
               + c3 / (mi) * np.sqrt(np.dot(divT.T, divT))
    cost = Jo + Jb + (np.dot(b1 * (constr1(grid, i) < 0), constr1(grid, i) ** 2) + np.dot(b2 * (constr2(grid, i) < 0),constr2(grid, i) ** 2)
                    + np.dot(b3 * (constr3(grid, i) < 0), constr3(grid, i) ** 2) + np.dot(b4 * (constr4(grid, i) < 0),constr4(grid, i) ** 2))


    # Derivative
    _, u_x, u_y = mapped_weight(u, y, x, yyT, xxT, i)
    dx = round(x[1]-x[0],2)
    dy = round(y[1]-y[0],2)

    if (err @ err) == 0:
        jac = np.zeros(len(grid))
    else:
        jac = - ((err @ err) ** (-1 / 2))  * np.sum(((err * (u_y/dy)).reshape((nt,nx*ny)) @ dydT) + ((err * (u_x/dx)).reshape((nt,nx*ny)) @ dxdT),axis=0)
    jac = jac / np.sqrt(nt0)

    if (np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) == 0 or c1 == 0:
        jac = jac
    else:
        jac2 = np.zeros(len(grid))
        jac2[0:mi ** 2] = xdif
        jac2[mi ** 2:mi ** 2 * 2] = ydif
        jac = jac + c1 /(mi)  * ((np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) ** (-1 / 2)) * jac2

    if (np.dot(dT.T, dT)) == 0 or c2 == 0:
        jac = jac
    else:
        jac3 = np.zeros(len(grid))
        jac3[0:mi ** 2] = Ax.T @ dT[0:mi ** 2] + Ay.T @ dT[mi ** 2:2 * mi ** 2]
        jac3[mi ** 2:mi ** 2 * 2] = Ax.T @ dT[2 * mi ** 2:3 * mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2]
        jac = jac + c2 /(mi)  * (np.dot(dT.T, dT) ** (-1 / 2)) * jac3

    if (np.dot(divT.T,divT)) == 0 or c3 == 0:
        jac = jac
    else:
        jac5 = np.zeros(len(grid))
        jac5[0:mi ** 2] = (Ax.T @ dT[0:mi ** 2] + Ax.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac5[mi ** 2:mi ** 2 * 2] =  (Ay.T @ dT[0:mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac = jac + c3 / (mi)  * (np.dot(divT.T,divT) ** (-1 / 2)) * jac5


    jac4 = 2 * (b1 * (constr1(grid,i) < 0) * constr1(grid,i)) @ dc1dX(grid,i) + 2 * (b2 * (constr2(grid,i) < 0) * constr2(grid,i)) @ dc2dX(grid,i) + 2 * (b3 * (constr3(grid,i) < 0) * constr3(grid,i)) @ dc3dX(grid,i) + 2 * (b4 * (constr4(grid,i) < 0) * constr4(grid,i)) @ dc4dX(grid,i)

    jac = jac + jac4


    return cost , jac




def Jp_a3(gridr, b, us, vs, y, x, i, c1, c2, c3, ct, dxdT, dydT, Ax, Ay, mask=None, Acomb=None, time_corr=None):
    # This function returns the value and the derivative of the cost function for the approach A3.
    # It takes as input:
    # - the two components of the mapping stored in gridr.
    # - the smoothed inputs us and vs.
    # - the spatial coordinates x and y.
    # - the step i (defining the smoothing and the resolution of the mapping).
    # - the regulation coefficients c1, c2, c3 and ct.
    # - the mask. It need to have the same dimension as the fields u and v. It is used to mask the area with no data (in case of irregularly spaced observations.
    # - Acomb is a matrix pairing two by two the time steps and time_corr the corresponding correlation.


    mi = 2**i + 1
    if len(us.shape) == 2:
        nt=1
    elif len(us.shape) == 3:
        nt = us.shape[0]

    ny = len(y)
    nx = len(x)

    grid = gridr.reshape((2*mi**2,nt))

    yyT = grid[(mi)**2:(mi)**2*2,:].reshape((mi,mi,nt))
    xxT = grid[0:(mi)**2,:].reshape((mi, mi,nt))

    v1 = vs
    u1 = mapped(us, y, x, yyT, xxT, i)

    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')
    ydif = np.zeros((mi**2,nt))
    xdif = np.zeros((mi**2,nt))
    for kt in range(nt):
        ydif[:,kt] = (yyT[:,:,kt]-yyc).reshape(-1)
        xdif[:,kt] = (xxT[:,:,kt]-xxc).reshape(-1)
    ydifr = ydif.reshape(-1)
    xdifr = xdif.reshape(-1)

    dT = np.zeros((4 * mi ** 2,nt))
    dT[0:mi ** 2,:] = Ax @ xdif
    dT[mi ** 2:2 * mi ** 2,:] = Ay @ xdif
    dT[2 * mi ** 2:3 * mi ** 2,:] = Ax @ ydif
    dT[3 * mi ** 2:4 * mi ** 2,:] = Ay @ ydif
    dTr = dT.reshape(-1)

    divT = Ax @ xdif + Ay @ ydif
    divTr = divT.reshape(-1)

    # Cost
    if mask is None:
        err = (v1 - u1).reshape(-1)
        Jo = np.sqrt(np.sum((v1 - u1) ** 2))
    else:
        err = (mask*(v1 - u1)).reshape(-1)
        Jo = np.sqrt((err @ err)) #Jo = np.sqrt(np.sum(mask * (v1 - u1) ** 2))



    Jb = c1 /(mi) * np.sqrt(np.dot(ydifr, ydifr) + np.dot(xdifr.T, xdifr)) \
               + c2 /(mi) * np.sqrt(np.dot(dTr.T, dTr)) \
               + c3 / (mi) * np.sqrt(np.dot(divTr.T, divTr))

    Cons1 = np.zeros(((mi-1)**2,nt))
    Cons2 = np.zeros(((mi-1)**2,nt))
    Cons3 = np.zeros(((mi-1)**2,nt))
    Cons4 = np.zeros(((mi-1)**2,nt))
    for kt in range(nt):
        Cons1[:, kt] = constr1(grid[:, kt], i)
        Cons2[:, kt] = constr2(grid[:, kt], i)
        Cons3[:, kt] = constr3(grid[:, kt], i)
        Cons4[:, kt] = constr4(grid[:, kt], i)
    Cons1r = Cons1.reshape(-1)
    Cons2r = Cons2.reshape(-1)
    Cons3r = Cons3.reshape(-1)
    Cons4r = Cons4.reshape(-1)


    Jc = (np.dot(b * (Cons1r < 0), Cons1r ** 2) + np.dot(b * (Cons2r < 0),Cons2r ** 2)
               + np.dot(b * (Cons3r < 0), Cons3r ** 2) + np.dot(b * (Cons4r < 0), Cons4r ** 2))
    #print(Jc)

    if (Acomb is not None) and (ct != 0):
        ydif_s = Acomb @ ydif.T
        xdif_s = Acomb @xdif.T
        #C = np.diag(np.sqrt(time_corr))
        # change begin
        C = np.diag(time_corr)
        # change end
        ydif_sc = (C @ ydif_s).reshape(-1)
        xdif_sc = (C @ xdif_s).reshape(-1)
        Js = ct  / mi* np.sqrt(ydif_sc.T @ ydif_sc + xdif_sc.T @ xdif_sc)
    else:
        Js = 0

    cost = Jo + Jb + Jc + Js

    # Derivative
    _, u_x, u_y = mapped_weight(us, y, x, yyT, xxT, i)
    dx = round(x[1]-x[0],2)
    dy = round(y[1]-y[0],2)

    if (err @ err) == 0:
        jac = np.zeros(grid.shape)
    else:
        jac = - ((err @ err) ** (-1 / 2))  * (((err * (u_y/dy)).reshape((nt,nx*ny)) @ dydT) + ((err * (u_x/dx)).reshape((nt,nx*ny)) @ dxdT)).T.reshape(-1)

        #jac = np.zeros(grid.shape)
        #for kt in range(nt):
        #jac = - ((np.sum((v1 - u1) ** 2)) ** (-1 / 2)) * ((((v1 - u1)).reshape(-1) * (u_y / dy) @ dydT) + (((v1 - u1)).reshape(-1) * (u_x / dx) @ dxdT))



    if (np.dot(ydifr.T, ydifr) + np.dot(xdifr.T, xdifr)) == 0 or c1 == 0:
        jac = jac
    else:
        jac2 = np.zeros(grid.shape)
        jac2[0:mi ** 2,:] = xdif
        jac2[mi ** 2:mi ** 2 * 2,:] = ydif
        jac2 = jac2.reshape(-1)
        jac = jac + c1 /(mi)  * ((np.dot(ydifr.T, ydifr) + np.dot(xdifr.T, xdifr)) ** (-1 / 2)) * jac2

    if (np.dot(dTr.T, dTr)) == 0 or c2 == 0:
        jac = jac
    else:
        jac3 = np.zeros(grid.shape)
        jac3[0:mi ** 2,:] = Ax.T @ dT[0:mi ** 2,:] + Ay.T @ dT[mi ** 2:2 * mi ** 2,:]
        jac3[mi ** 2:mi ** 2 * 2,:] = Ax.T @ dT[2 * mi ** 2:3 * mi ** 2,:] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2,:]
        jac3 = jac3.reshape(-1)
        jac = jac + c2 /(mi)  * (np.dot(dTr.T, dTr) ** (-1 / 2)) * jac3

    if (np.dot(divTr.T,divTr)) == 0 or c3 == 0:
        jac = jac
    else:
        jac5 = np.zeros(grid.shape)
        jac5[0:mi ** 2,:] = (Ax.T @ dT[0:mi ** 2,:] + Ax.T @ dT[3 * mi ** 2:4 * mi ** 2,:])
        jac5[mi ** 2:mi ** 2 * 2,:] =  (Ay.T @ dT[0:mi ** 2,:] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2,:])
        jac5 = jac5.reshape(-1)
        jac = jac + c3 / (mi)  * (np.dot(divTr.T,divTr) ** (-1 / 2)) * jac5

    if (Acomb is not None) and (c4!=0):
        if ((xdif_sc.T @ xdif_sc)==0 and  (ydif_sc.T @ ydif_sc)==0):
            jac = jac
        else:
            #change begin
            C = np.diag(time_corr)
            #change end
            temp = np.zeros(grid.shape)
            B = Acomb.T @ C.T @ C @ Acomb
            temp[0:(mi) ** 2, :] = xdif @ B
            temp[(mi) ** 2:(mi) ** 2 * 2, :] = ydif @ B
            jac6 = c4 /mi * (ydif_sc.T @ ydif_sc + xdif_sc.T @ xdif_sc) ** (-1 / 2) * temp
            jac = jac + jac6.reshape(-1)
    else:
        jac = jac



    DCons1 = np.zeros((2 * mi ** 2, nt))
    DCons2 = np.zeros((2 * mi ** 2, nt))
    DCons3 = np.zeros((2 * mi ** 2, nt))
    DCons4 = np.zeros((2 * mi ** 2, nt))
    for kt in range(nt):
        DCons1[:, kt] =  2 * (b * (Cons1 < 0) * Cons1)[:,kt] @  dc1dX(grid[:, kt], i)
        DCons2[:, kt] =  2 * (b * (Cons2 < 0) * Cons2)[:,kt] @  dc2dX(grid[:, kt], i)
        DCons3[:, kt] =  2 * (b * (Cons3 < 0) * Cons3)[:,kt] @  dc3dX(grid[:, kt], i)
        DCons4[:, kt] =  2 * (b * (Cons4 < 0) * Cons4)[:,kt] @  dc4dX(grid[:, kt], i)

    jac4  = (DCons1 + DCons2 + DCons3 + DCons4).reshape(-1)

    jac = jac + jac4


    return cost , jac



def Jp2D(grid,b,u,v,y,x,i,c1,c2,c3,dxdT,dydT,Ax,Ay,mask=None):
    # Penalized cost function Jp with derivative
    mi = 2**i + 1

    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')
    yyT = grid[(mi)**2:(mi)**2*2].reshape((mi,mi))
    xxT = grid[0:(mi)**2].reshape((mi, mi))
    b1 = b #[0 : (mi-1)**2]
    b2 = b #[(mi-1)**2 : 2*(mi - 1)**2]
    b3 = b #[2*(mi-1)**2 : 3*(mi - 1)**2]
    b4 = b #[3*(mi-1)**2 :  4*(mi - 1)**2]

    v1 = v
    u1 = mapped(u, y, x, yyT, xxT, i)

    ydif = (yyT-yyc).reshape(-1)
    xdif = (xxT-xxc).reshape(-1)

    dT = np.zeros(4 * mi ** 2)
    dT[0:mi ** 2] = Ax @ xdif
    dT[mi ** 2:2 * mi ** 2] = Ay @ xdif
    dT[2 * mi ** 2:3 * mi ** 2] = Ax @ ydif
    dT[3 * mi ** 2:4 * mi ** 2] = Ay @ ydif

    divT = Ax @ xdif + Ay @ ydif

    # Cost
    if mask is None:
        Jo = np.sqrt(sum(sum((v1 - u1) ** 2)))
    else:
        Jo = np.sqrt(sum(sum(mask * (v1 - u1) ** 2)))

    Jb = c1 /(mi) * np.sqrt(np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) \
               + c2 /(mi) * np.sqrt(np.dot(dT.T, dT)) \
               + c3 / (mi) * np.sqrt(np.dot(divT.T, divT))
    cost = Jo + Jb + (np.dot(b1 * (constr1(grid, i) < 0), constr1(grid, i) ** 2) + np.dot(b2 * (constr2(grid, i) < 0),constr2(grid, i) ** 2)
                    + np.dot(b3 * (constr3(grid, i) < 0), constr3(grid, i) ** 2) + np.dot(b4 * (constr4(grid, i) < 0),constr4(grid, i) ** 2))


    # Derivative
    u_w, u_x, u_y = mapped_weight(u, y, x, yyT, xxT, i)
    dx = round(x[1]-x[0],2)
    dy = round(y[1]-y[0],2)

    if mask is None:
        #jac = - ((sum(sum((v1 - u1) ** 2))) ** (-1 / 2)) * ((((v1 - u1)).reshape(-1) * (u_y/dy) @ dydT) + (((v1 - u1)).reshape(-1) * (u_x/dx) @ dxdT))
        jac = - ((np.sum((v1 - u1) ** 2)) ** (-1 / 2)) * (
                    (((v1 - u1)).reshape(-1) * (u_y / dy) @ dydT) + (((v1 - u1)).reshape(-1) * (u_x / dx) @ dxdT))

    else:
        jac = - ((sum(sum(mask*(v1 - u1) ** 2))) ** (-1 / 2)) * (((mask*(v1 - u1)).reshape(-1) * (u_y / dy) @ dydT) + ((mask*(v1 - u1)).reshape(-1) * (u_x / dx) @ dxdT))

    if (np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) == 0 or c1 == 0:
        jac = jac
    else:
        jac2 = np.zeros(len(grid))
        jac2[0:mi ** 2] = xdif
        jac2[mi ** 2:mi ** 2 * 2] = ydif
        jac = jac + c1 /(mi)  * ((np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) ** (-1 / 2)) * jac2

    if (np.dot(dT.T, dT)) == 0 or c2 == 0:
        jac = jac
    else:
        jac3 = np.zeros(len(grid))
        jac3[0:mi ** 2] = Ax.T @ dT[0:mi ** 2] + Ay.T @ dT[mi ** 2:2 * mi ** 2]
        jac3[mi ** 2:mi ** 2 * 2] = Ax.T @ dT[2 * mi ** 2:3 * mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2]
        jac = jac + c2 /(mi)  * (np.dot(dT.T, dT) ** (-1 / 2)) * jac3

    if (np.dot(divT.T,divT)) == 0 or c3 == 0:
        jac = jac
    else:
        jac5 = np.zeros(len(grid))
        jac5[0:mi ** 2] = (Ax.T @ dT[0:mi ** 2] + Ax.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac5[mi ** 2:mi ** 2 * 2] =  (Ay.T @ dT[0:mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac = jac + c3 / (mi)  * (np.dot(divT.T,divT) ** (-1 / 2)) * jac5


    jac4 = 2 * (b1 * (constr1(grid,i) < 0) * constr1(grid,i)) @ dc1dX(grid,i) + 2 * (b2 * (constr2(grid,i) < 0) * constr2(grid,i)) @ dc2dX(grid,i) + 2 * (b3 * (constr3(grid,i) < 0) * constr3(grid,i)) @ dc3dX(grid,i) + 2 * (b4 * (constr4(grid,i) < 0) * constr4(grid,i)) @ dc4dX(grid,i)

    jac = jac + jac4

    return cost, jac