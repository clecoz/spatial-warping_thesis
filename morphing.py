import numpy as np
from scipy import interpolate

################################################################################
#
# Morphing and related transformation
#
###############################################################################


def morphing(u, v, y, x, yyT, xxT, i, la):
    # Morph field u according to mapping T (ttT,xxT) and lambda (la)
    # u: field to morph
    # v: target field
    # t,x : coordinate of pixel grid
    # ttT, xxT: mapping T
    # i: number of the morphing grid on which T is given
    # lambda: morphing coefficient

    ny = len(y)
    nx = len(x)
    if len(u.shape) == 2:
        nt = 1
    elif len(u.shape) == 3:
        nt = u.shape[0]
    yc = range(0, ny, int((ny - 1) / 2 ** i))
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')

    if len(yyT.shape) == 2:
        # Transform coordinate
        Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear', bounds_error=False, fill_value=None)
        tt_prime = Tt.reshape((nx, ny))
        Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear',bounds_error=False, fill_value=None)
        xx_prime = Tx.reshape((nx, ny))
        # morphed coordinates
        xxl = xx + la * (xx_prime - xx)
        ttl = yy + la * (tt_prime - yy)
        # Morphed signal (and v inverse transform)
        points = np.array([xxT.reshape(-1), yyT.reshape(-1)]).T
        values = xxc.reshape(-1)
        xxT_inv = interpolate.griddata(points, values, (xx, yy), method='linear')
        yyT_inv = interpolate.griddata(points, yyc.reshape(-1), (xx, yy), method='linear')

        u_morph = np.zeros(u.shape)
        if len(u.shape) == 2:
            v_inv = interpolate.interpn((x, y), v,np.array([xxT_inv.reshape(-1), yyT_inv.reshape(-1)]).T, method='linear',
                                        bounds_error=False, fill_value=0).reshape((nx, ny))
            v_inv[np.isnan(v_inv)] = 0
            u_morph = interpolate.interpn((x, y), (1.0 - la) * u + la * v_inv,np.array([xxl.reshape(-1), ttl.reshape(-1)]).T,
                                                    method='linear', bounds_error=False, fill_value=0).reshape((nx, ny))

        elif len(u.shape) == 3:
            for kt in range(nt):
                v_inv = interpolate.interpn((x, y), np.squeeze(v[kt,:,:]), np.array([xxT_inv.reshape(-1), yyT_inv.reshape(-1)]).T, method='linear',
                                        bounds_error=False, fill_value=0).reshape((nx, ny))
                v_inv[np.isnan(v_inv)] = 0
                u_morph[kt,:,:] = interpolate.interpn((x, y), (1.0 - la) * u[kt,:,:] + la * v_inv, np.array([xxl.reshape(-1), ttl.reshape(-1)]).T,
                                          method='linear', bounds_error=False, fill_value=0).reshape((nx, ny))

    elif len(yyT.shape) == 3:
        u_morph = np.zeros(u.shape)
        for kt in range(nt):
            # Transform coordinate
            Tt = interpolate.interpn((x[xc], y[yc]), yyT[:,:,kt], np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear', bounds_error=False, fill_value=None)
            tt_prime = Tt.reshape((nx, ny))
            Tx = interpolate.interpn((x[xc], y[yc]), xxT[:,:,kt], np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear', bounds_error=False, fill_value=None)
            xx_prime = Tx.reshape((nx, ny))
            # morphed coordinates
            xxl = xx + la * (xx_prime - xx)
            ttl = yy + la * (tt_prime - yy)
            # Morphed signal (and v inverse transform)
            points = np.array([xxT[:,:,kt].reshape(-1), yyT[:,:,kt].reshape(-1)]).T
            values = xxc.reshape(-1)
            xxT_inv = interpolate.griddata(points, values, (xx, yy), method='linear')
            yyT_inv = interpolate.griddata(points, yyc.reshape(-1), (xx, yy), method='linear')

            v_inv = interpolate.interpn((x, y), v[kt,:,:], np.array([xxT_inv.reshape(-1), yyT_inv.reshape(-1)]).T,method='linear',bounds_error=False, fill_value=0).reshape((nx, ny))
            v_inv[np.isnan(v_inv)] = 0
            u_morph = interpolate.interpn((x, y), (1.0 - la) * u[kt,:,:] + la * v_inv,np.array([xxl.reshape(-1), ttl.reshape(-1)]).T, method='linear', bounds_error=False, fill_value=0).reshape((nx, ny))

    return u_morph

def morph_res(u, v, t, x, ttT, xxT, i):
    nt = len(t)
    nx = len(x)
    #tc = range(0, nt, int((nt - 1) / 2 ** i))
    #xc = range(0, nx, int((nx - 1) / 2 ** i))
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xxc, ttc = np.meshgrid(xc, tc, indexing='ij')

    # Transform coordinate
    Tt = interpolate.interpn((xc, tc), ttT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear',
                             bounds_error=False, fill_value=None)
    tt_prime = Tt.reshape((nx, nt))
    Tx = interpolate.interpn((xc, tc), xxT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear',
                             bounds_error=False, fill_value=None)
    xx_prime = Tx.reshape((nx, nt))

    # v inverse transform
    points = np.array([xxT.reshape(-1), ttT.reshape(-1)]).T
    values = xxc.reshape(-1)
    xxT_inv = interpolate.griddata(points, values, (xx, tt), method='linear')
    ttT_inv = interpolate.griddata(points, ttc.reshape(-1), (xx, tt), method='linear')
    v_inv = interpolate.interpn((x, t), v, np.array([xxT_inv.reshape(-1), ttT_inv.reshape(-1)]).T, method='linear',
                                bounds_error=False, fill_value=0)
    v_inv = v_inv.reshape((nx, nt))
    v_inv[np.isnan(v_inv)] = 0

    return (v_inv).reshape(nx, nt)-u

def mapped_spline(u,y,x,yyT,xxT,i):
    # Return the warped signal
    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Transform coordinate
    Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
    Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')

    # Interpolated function
    uT = interpolate.interpn((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=100)  # ,fill_value=1000)
    return uT.reshape(nx,ny)



def distort_grid(y,x,yyT, xxT,i, la):
    ny = len(y)
    nx = len(x)
    mi = 2**i+1
    xx, yy = np.meshgrid(x, y, indexing='ij')
    yc = np.linspace(0, ny - 1, mi, dtype=int)
    xc = np.linspace(0, nx - 1, mi, dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')

    if len(yyT.shape) == 2:
        nt = 1
    elif len(yyT.shape) == 3:
        nt = yyT.shape[-1]

    if nt == 1:
        # Inverse transform coordinates
        points = np.array([(xxc + la * (xxT - xxc)).reshape(-1), (yyc + la*(yyT-yyc)).reshape(-1)]).T
        xx_l = interpolate.griddata(points, xxc.reshape(-1), (xx, yy), method='linear').reshape(nx,ny)
        yy_l = interpolate.griddata(points, yyc.reshape(-1), (xx, yy), method='linear').reshape(nx,ny)
    else:
        xx_l = np.zeros((nx,ny,nt))
        yy_l = np.zeros((nx, ny, nt))
        for kt in range(nt):
            points = np.array([(xxc + la * (xxT[:,:,kt] - xxc)).reshape(-1), (yyc + la * (yyT[:,:,kt] - yyc)).reshape(-1)]).T
            xx_l[:,:,kt] = interpolate.griddata(points, xxc.reshape(-1), (xx, yy), method='linear').reshape(nx, ny)
            yy_l[:,:,kt] = interpolate.griddata(points, yyc.reshape(-1), (xx, yy), method='linear').reshape(nx, ny)

    return xx_l, yy_l
