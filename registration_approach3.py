import scipy.optimize as op
import time
from cost_function import *
from scipy.sparse import csr_matrix

#=============================================


def registration_a3(u,v,x,y,I,c1,c2,c3,ct,mask,Acomb,time_cor,folder_results,ks):
    # This function performs the automatic registration and return the mappings.
    # It takes as input:
    # - u, the rainfall field to be corrected.
    # - v, the target rainfall field (assumed to be the truth). The fields u and v need to have the same dimensions.
    # - the corresponding coordinates, x and y being the longitude and latitude respectively.
    # - the number of steps I (corresponding to the number of morphing grid). I has to be an integer.
    # - the regulation coefficients c1, c2, c3 and ct (floats).
    # - the mask. It needs to have the same dimension as the fields u and v. It is used to mask the area with no data (in case of irregularly spaced observations.
    # - Acomb is a matrix pairing two by two the time steps and time_corr the corresponding correlation. Together, they define the influence function.
    # - folder_results is a folder, the mappings will be saved in this folder for future use.
    # - ks is added to the file names containing the mappings (used for the LOOV experiment).

    nx = len(x)
    ny = len(y)
    nt,_,_ = v.shape

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
            xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')
            mxT, myT = np.meshgrid(x[xc], y[yc], indexing='ij')

            grid2 = np.zeros(((2 ** i + 1) ** 2 * 2, nt))
            for kt in range(nt):
                grid2[0:(2 ** i + 1) ** 2, kt] = mxT.reshape(-1)
                grid2[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2, kt] = myT.reshape(-1)
        else:
            # New undistorted finer grid
            yc_new = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
            xc_new = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
            xxc, yyc = np.meshgrid(x[xc_new], y[yc_new], indexing='ij')

            grid2 = np.zeros(((2 ** i + 1) ** 2 * 2,nt))
            for kt in range(nt):
                # Interpolate T from coarser grid into the new grid
                Ty = interpolate.interpn((x[xc], y[yc]), myT[:,:,kt], np.array([xxc.reshape(-1), yyc.reshape(-1)]).T,method='linear')
                Tx = interpolate.interpn((x[xc], y[yc]), mxT[:,:,kt], np.array([xxc.reshape(-1), yyc.reshape(-1)]).T,method='linear')
                # Initialize
                grid2[0:(2 ** i + 1) ** 2, kt] = Tx.reshape(-1)
                grid2[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2, kt] = Ty.reshape(-1)

            # Reset grid
            xc = xc_new
            yc = yc_new

        grid = grid2.reshape(-1)


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
        bnd_xmin = - np.inf * np.ones((mi,mi,nt))
        bnd_xmax = np.inf * np.ones((mi,mi,nt))
        bnd_ymin = - np.inf * np.ones((mi,mi,nt))
        bnd_ymax = np.inf * np.ones((mi,mi,nt))
        bnd_xmin[-1, :,:] = xmax
        bnd_xmax[0, :,:] = xmin
        bnd_ymin[:, -1,:] = ymax
        bnd_ymax[:, 0,:] = ymin
        bnd_xmin = bnd_xmin.reshape(-1)
        bnd_xmax = bnd_xmax.reshape(-1)
        bnd_ymin = bnd_ymin.reshape(-1)
        bnd_ymax = bnd_ymax.reshape(-1)

        for k in range(0, len(bnd_xmin)):
            bnds = bnds + ((bnd_xmin[k], bnd_xmax[k]),)
        for k in range(len(bnd_xmin),len(bnd_xmin)+len(bnd_ymin)):
            bnds = bnds + ((bnd_ymin[k-len(bnd_xmin)], bnd_ymax[k-len(bnd_xmin)]),)


        # Smooth signals
        vs = smooth(v, y, x, i)
        us = smooth(u, y, x, i)

        # Normalize precipitation
        for kt in range(nt):
            if np.max(us[kt,:,:]) != 0:
                us[kt,:,:] = np.max(vs[kt,:,:]) * us[kt,:,:] / np.max(us[kt,:,:])


        # Initialize beta and stopping criteria
        b = 1
        crit1 = 1
        crit2 = 1

        while (crit1>eps1 or crit2>eps2) :
            # Optimization (with derivative)
            tTo1 = op.minimize(Jp_a3, grid, args=(b, us, vs, y, x, i, c1, c2, c3,ct, dxdT, dydT, Ax, At, mask,Acomb,time_cor), jac=True,
                               method='L-BFGS-B', bounds=bnds, options={'maxiter': 10000, 'maxfun': 100000})
            print('Optimization successful: {}'.format(tTo1.success))  # did the minimization succeed?

            # Check if constrains are respected
            gridr = tTo1.x.reshape((2 * mi ** 2, nt))
            Cons1 = np.zeros(((mi - 1) ** 2, nt))
            Cons2 = np.zeros(((mi - 1) ** 2, nt))
            Cons3 = np.zeros(((mi - 1) ** 2, nt))
            Cons4 = np.zeros(((mi - 1) ** 2, nt))
            for kt in range(nt):
                Cons1[:, kt] = constr1(gridr[:, kt], i)
                Cons2[:, kt] = constr2(gridr[:, kt], i)
                Cons3[:, kt] = constr3(gridr[:, kt], i)
                Cons4[:, kt] = constr4(gridr[:, kt], i)
            if np.all(Cons1 >= 0) and np.all(Cons2 >= 0) and np.all(Cons3 >= 0) and np.all(Cons4 >= 0):
                crit1 = 0
                crit2 = 0
            else:
                crit1 = np.sqrt(np.sum((grid-tTo1.x)**2))
                crit2 = np.abs(Jp_a3(grid, b, us, vs, y, x, i,c1,c2,c3,ct, dxdT, dydT, Ax, At,mask)[0] - Jp_a3(tTo1.x, b, us, vs, y, x, i,c1,c2,c3,ct, dxdT, dydT, Ax, At,mask)[0])

            # Update
            grid = tTo1.x
            b = 10*b


        # Update distorted grid
        gridr = grid.reshape((2 * mi ** 2, nt))
        myT = gridr[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2, :].reshape((2 ** i + 1, 2 ** i + 1, nt))
        mxT = gridr[0:(2 ** i + 1) ** 2, :].reshape((2 ** i + 1, 2 ** i + 1, nt))


        # Save mapping for future use
        np.savetxt(folder_results + '/mxT_i{}_ks{}.csv'.format(i,ks), mxT.reshape((mi**2,nt)), delimiter=',')
        np.savetxt(folder_results + '/myT_i{}_ks{}.csv'.format(i,ks),myT.reshape((mi**2,nt)), delimiter=',')

        # Print elapsed time
        end = time.time()
        print('Elapsed time for ks {}: {}'.format(ks,end - start))

    # Returm mappings
    return mxT, myT