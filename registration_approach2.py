import scipy.optimize as op
import time
from cost_function import *
from scipy.sparse import csr_matrix

#=============================================

def registration_a2(u,v,x,y,I,c1,c2,c3,mask,folder_results,ks):
    # This function perform the automatic registration and return the mappings.
    # It takes as input:
    # - u, the rainfall field to be corrected.
    # - v, the target rainfall field (assumed to be the truth). The fields u and v need to have the same dimensions.
    # - the corresponding coordinates, x and y being the longitude and latitude respectively.
    # - the number of steps I (corresponding to the number of morphing grid). I has to be an integer.
    # - the regulation coefficients c1, c2 and c3 (floats).
    # - the mask. It need to have the same dimension as the fields u and v. It is used to mask the area with no data (in case of irregularly spaced observations.
    # - folder_results is a folder, the mappings will be saved in this folder for future use.
    # - ks is added to the file names containing the mappings (used for the LOOV experiment).

    nx = len(x)
    ny = len(y)
    nt, _, _ = v.shape
    nt0 = np.sum((u.reshape((nt, nx * ny)) > 0.1).any(axis=1))

    # Stopping criteria for the iterative barrier approach
    eps1 = 10**-5
    eps2 = 10**-5

    # Iteration on the morphing grids (i<=I)
    for i in range(1,I+1):
        print('Step i='+str(i))
        start = time.time()
        mi = 2 ** i + 1

        # Initialization
        if i==1:
            # Undistorted coarser grid
            yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
            xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
            mxT, myT = np.meshgrid(x[xc], y[yc], indexing='ij')
            xxc, ttc = np.meshgrid(x[xc], y[yc], indexing='ij')

            grid = np.zeros((2 ** i + 1) ** 2 * 2 )
            grid[0:(2 ** i + 1) ** 2] = mxT.reshape(-1)
            grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2] = myT.reshape(-1)
        else:
            # New undistorted finer grid
            yc_new = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
            xc_new = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)

            # Interpolate T from coarser grid into the new grid
            xxc, ttc = np.meshgrid(x[xc_new], y[yc_new], indexing='ij')
            Ty = interpolate.interpn((x[xc], y[yc]), myT, np.array([xxc.reshape(-1), ttc.reshape(-1)]).T, method='linear')
            Tx = interpolate.interpn((x[xc], y[yc]), mxT, np.array([xxc.reshape(-1), ttc.reshape(-1)]).T, method='linear')
            # Initialize
            myT = Ty.reshape((mi, mi))
            mxT = Tx.reshape((mi, mi))
            grid = np.zeros((2 ** i + 1) ** 2 * 2)
            grid[0:(2 ** i + 1) ** 2] = mxT.reshape(-1)
            grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2] = myT.reshape(-1)

            # Reset grid
            xc = xc_new
            yc = yc_new

        #==========================================================================
        # Pre-computation of derivative elements non-dependant of the distortion
        dxdT, dydT = dXdT(y, x, i)
        dxdT = csr_matrix(dxdT)
        dydT = csr_matrix(dydT)
        # Derivation matrix
        Ax = np.diag(-1 * np.ones(mi * (mi - 1)), -mi) + np.diag(np.ones(mi * (mi - 1)), mi)
        Ax[0:mi, 0:mi] = -1 * np.eye(mi)
        Ax[mi * (mi - 1):mi ** 2, mi * (mi - 1):mi ** 2] = 1 * np.eye(mi)
        Ax[0:mi, :] = 1 / (xc[1] - xc[0]) * Ax[0:mi, :]
        Ax[mi * (mi - 1):mi ** 2, :] = 1 / (xc[1] - xc[0]) * Ax[mi * (mi - 1):mi ** 2, :]
        Ax[mi:mi * (mi - 1), :] = 1 / (xc[2] - xc[0]) * Ax[mi:mi * (mi - 1), :]
        at = np.diag(-1 * np.ones(mi - 1), -1) + np.diag(np.ones(mi - 1), 1)
        at[0, 0] = -1
        at[mi - 1, mi - 1] = 1
        at[0, :] = 1 / (yc[1] - yc[0]) * at[0, :]
        at[mi - 1, :] = 1 / (yc[1] - yc[0]) * at[mi - 1, :]
        at[1:mi - 1, :] = 1 / (yc[2] - yc[0]) * at[1:mi - 1, :]
        temp = np.multiply.outer(np.eye(mi), at)
        At = np.swapaxes(temp, 1, 2).reshape((mi ** 2, mi ** 2))
        Ax = csr_matrix(Ax)
        At = csr_matrix(At)

        #==========================================================================
        # Define bounds
        bnds = ()
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        bnd_xmin = - np.inf * np.ones(mxT.shape)
        bnd_xmax = np.inf * np.ones(mxT.shape)
        bnd_ymin = - np.inf * np.ones(myT.shape)
        bnd_ymax = np.inf * np.ones(myT.shape)
        bnd_xmin[-1, :] = xmax
        bnd_xmax[0, :] = xmin
        bnd_ymin[:, -1] = ymax
        bnd_ymax[:, 0] = ymin
        bnd_xmin = bnd_xmin.reshape(-1)
        bnd_xmax = bnd_xmax.reshape(-1)
        bnd_ymin = bnd_ymin.reshape(-1)
        bnd_ymax = bnd_ymax.reshape(-1)

        for k in range(0, mi ** 2):
            bnds = bnds + ((bnd_xmin[k], bnd_xmax[k]),)
        for k in range(mi ** 2, 2 * mi ** 2):
            bnds = bnds + ((bnd_ymin[k-mi**2], bnd_ymax[k-mi**2]),)

        # Smooth signals
        vs = smooth(v, y, x, i)
        us = smooth(u, y, x, i)

        # Normalize precipitation
        for kt in range(nt):
            if np.max(us[kt,:,:]) != 0:
                us[kt, :, :] = np.max(vs[kt, :, :]) * us[kt, :, :] / np.max(us[kt, :, :])

        # Initialize beta and stopping criteria
        b = 1*np.ones(4*(mi - 1)**2)
        crit1 = 1
        crit2 = 1

        while (crit1>eps1 or crit2>eps2) :
            # without derivative
            #tTo1 = op.minimize(Jp, grid, args=(b,us, vs, y, x, i,c1,c2,c3,dxdT, dydT, Ax, At,mask), jac=False, method='L-BFGS-B',bounds=bnds,options={'maxiter': 100,'maxfun':1000000})
            # with derivative
            tTo1 = op.minimize(Jp, grid, args=(b, us, vs, y, x, i,c1,c2,c3,dxdT, dydT, Ax, At,mask,nt0), jac=True, method='L-BFGS-B', bounds=bnds,options={'maxiter':10000, 'maxfun': 100000,'maxls':20})
            print('Optimization successful: {}'.format(tTo1.success))    # did the minimization succeed?


            # Check if constrains are respected
            if np.all(constr1(tTo1.x,i)>=0) and np.all(constr2(tTo1.x,i)>=0) and np.all(constr3(tTo1.x,i)>=0) and np.all(constr4(tTo1.x,i)>=0):
                crit1 = 0
                crit2 = 0
            else:
                crit1 = np.sqrt(np.sum((grid-tTo1.x)**2))
                crit2 = np.abs(Jp(grid, b, us, vs, y, x, i,c1,c2,c3, dxdT, dydT, Ax, At,mask)[0] - Jp(tTo1.x, b, us, vs, y, x, i,c1,c2,c3, dxdT, dydT, Ax, At,mask)[0])

            # Update
            grid = tTo1.x
            b = 10*b

        # Update distorted grid
        myT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
        mxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))

        # Save mapping for future use
        np.savetxt(folder_results + '/mxT_i{}_ks{}.csv'.format(i, ks), mxT, delimiter=',')
        np.savetxt(folder_results + '/myT_i{}_ks{}.csv'.format(i, ks), myT, delimiter=',')

        # Print elapsed time
        end = time.time()
        print('Elapsed time for step {}: {}'.format(i,end - start))

    # Return mapping
    return mxT, myT
