from registration_approach3 import *
from registration_approach2 import *
from registration_approach1 import *
from pykrige import OrdinaryKriging
import pandas as pd
import numpy as np
import os
import matplotlib.colors as cls
from matplotlib.colors import ListedColormap
from itertools import combinations
import math
from mpl_toolkits.basemap import Basemap, cm

#=======================================================================================================================
# Synthetic case
#=======================================================================================================================

# Choose case
case = 'case_spacetime'

# Choose experience
exp = "full"    #"full" or "interpolated"

# Choose where to save results
folder_result = "{}/results/".format(case).replace('.','p')
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

# Choose if you want to plot or compute statistics
plot = True
stats = True
threshold = 20   # in mm/h, used to compute the position and timing error

#==================================================================
# Choose registration parameter

# Choose registration approach
reg = "a2"          # Possible choice: "a1", "a2" and "a3"

# Choose regulation coefficients
c1 = 0.1
c2 = 1
c3 = 1
c4 = 1
c5 = 0.3            # only needed for approach a3

# Choose number of morphing grids
I = 4


#==================================================================
# Prepare data from the chosen case
print('Prepare data')

# Files containing the data
TAHMO_file = './{}/v.csv'.format(case)
TAHMO_sta_file = './{}/v_stations.csv'.format(case)
IMERG_file = './{}/u.csv'.format(case)
file_coord_sta = './coord_stations.csv'

# Dimensions
nt = 25
nx = 65
ny = 65

# Data to be corrected (IMERG-Late)
u_init = np.loadtxt(IMERG_file, delimiter=',').reshape((25,65,65))
u_input = u_init[:,14:-14,14:-14]             # We select a subdomain of 37 by 37 grid points. The goal is to make this case similar to the southern Ghana case (in term of size)
lon_u = np.array(np.arange(-4.75, 1.75, 0.1))
lat_u = np.array(np.arange(3.25, 9.75, 0.1))
npts = lon_u.shape[0]

# Reference data (TAHMO)
df_coord_sta = pd.read_csv(file_coord_sta, index_col=0)
lon_v = df_coord_sta.loc['lon'].values
lat_v = df_coord_sta.loc['lat'].values
v_truth = np.loadtxt(TAHMO_file, delimiter=',').reshape((-1,65,65))
v_station = np.loadtxt(TAHMO_sta_file, delimiter=',')

# Coordinates
t = np.array(range(0, nt))
x = lon_u
y = lat_u
nx = len(x)
ny = len(y)



if exp == "interpolated":
    # Interpolation of (TAHMO) stations on the regular grid
    v_input = np.zeros((nt,37,37))
    for k in range(nt):
        OK = OrdinaryKriging(lon_v, lat_v, np.sqrt(v_station[k, :]), variogram_model='exponential',
                             variogram_parameters={'sill': 1.0, 'range': 2, 'nugget': 0.01}, nlags=50, verbose=False,
                             enable_plotting=False, weight=True, coordinates_type='geographic')
        z, ss = OK.execute('grid', lon_u[14:-14], lat_u[14:-14])
        v_input[k, :, :] = (z ** 2).T

elif exp == "full":
    # For the "full" experiment, the reference data is already on the grid. We only have to select the domain we want (37 by 37 grid points)
    v_input = v_truth[:,14:-14,14:-14]



#==================================================================
# Pre-processing
print('\nStart pre-processing')

# Extend the domain
u = np.zeros((nt, nx, ny))
u[:, 14:-14, 14:-14] = u_input
v = np.zeros((nt, nx, ny))
v[:, 14:-14, 14:-14] = v_input

# Remove light precipitation (<1mm)
u[u < 1] = 0
v[v < 1] = 0


#================================================================
# Define the correlation between time steps (saved in a matrix Acomb)

Acomb = np.zeros((len(list(combinations(range(nt), 2))),nt))
time_cor = np.zeros(len(list(combinations(range(nt), 2))))
nit = 0
for k, j in combinations(range(nt), 2):
    Acomb[nit,k] = -1
    Acomb[nit, j] = 1
    if np.abs(k-j) == 1:
        time_cor[nit] = c5
    nit += 1


#================================================================
# Define mask
mask = np.zeros((nt, nx, ny))
if exp == "interpolated":
    for kt in range(nt):
        mask[kt, 14:-14, 14:-14] = (ss < 0.5).astype("float").T
elif exp == "full":
    mask[:, 14:-14, 14:-14] = 1


#================================================================
# Registration
print("\nStart regisration")

ks = 'all'          # this parameter is used for the cross-validation in the case of the southern Ghana case

start = time.time()

if reg == "a1":
    Tx, Ty = registration_a1(u, v, x, y, I, c1, c2, c3, mask, folder_result, ks)
elif reg == "a2":
    Tx, Ty = registration_a2(u, v, x, y, I, c1, c2, c3, mask, folder_result, ks)
elif reg == "a3":
    Tx, Ty = registration_a3(u, v, x, y, I, c1, c2, c3, c4, mask, Acomb, time_cor, folder_result, ks)
else:
    print("Error (wrong ""reg""): this approach does not exist.")


end = time.time()
print('\n Elapsed time for ks {}: {}'.format(ks,end - start))


#================================================================
# Warping
print("\nWarping")

u_warped = mapped(u_init, y, x, Ty, Tx, I)


#================================================================
# Statistics

if stats:
    print('\nStatistics')
    # The statistics are computed on the original domain not the extended one
    u_w2 = u_warped[:, 14:-14, 14:-14]
    v2 = v_truth[:, 14:-14, 14:-14]
    x2 = lon_u[14:-14]
    y2 = lat_u[14:-14]

    # Statistics before warping
    RMSE_before = np.sqrt(np.mean((u_input - v2) ** 2))
    MAE_before = np.mean(np.abs(u_input - v2))
    C_before = np.corrcoef(u_input.reshape(-1), v2.reshape(-1))[0, 1]

    # Statistics after warping
    RMSE_after = np.sqrt(np.mean((u_w2 - v2) ** 2))
    MAE_after = np.mean(np.abs(u_w2 - v2))
    C_after = np.corrcoef(u_w2.reshape(-1), v2.reshape(-1))[0, 1]

    print('#---------------------------------------#')
    print('#        |  RMSE  |  MAE  | Correlation #')
    print('#---------------------------------------#')
    print('# Before |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_before,MAE_before,C_before))
    print('# After  |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_after,MAE_after,C_after))
    print('#---------------------------------------#')



    #---------------------------------------------
    # Position error
    # Find indexes of the pixel with the max. rain
    iu, ju = np.unravel_index(np.argmax(u_input.reshape(nt, -1), axis=1), (37, 37))
    iv, jv = np.unravel_index(np.argmax(v2.reshape(nt, -1), axis=1), (37, 37))
    iuw, juw = np.unravel_index(np.argmax(u_w2.reshape(nt, -1), axis=1), (37, 37))

    # Compute distance between max.
    def distance(origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371  # km
        dlat = math.radians(lat2) - math.radians(lat1)
        dlon = math.radians(lon2) - math.radians(lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.asin(np.sqrt(a))
        d = radius * c
        return d

    dist_before = np.zeros(nt)
    dist_after = np.zeros(nt)
    for kt in range(nt):
        dist_before[kt] = distance((y2[ju[kt]], x2[iu[kt]]), (y2[jv[kt]], x2[iv[kt]]))
        dist_after[kt] = distance((y2[juw[kt]], x2[iuw[kt]]), (y2[jv[kt]], x2[iv[kt]]))

    # Compute average position error
    mask = (~(v2 < threshold).all(axis=(1, 2)) * ~(u_input < 1).all(axis=(1, 2)))
    average_dist_before = np.mean(dist_before[mask])
    average_dist_after = np.mean(dist_after[mask])

    # Print results
    print("\nAverage position error for threshold={}mm/h (sample number = {})".format(threshold,np.sum(mask)))
    print("Before: {:.2f} km".format(average_dist_before))
    print("After:  {:.2f} km".format(average_dist_after))


    # ---------------------------------------------
    # Timing error
    # Find indexes of the pixel with the max. rain
    tv = np.argmax(v2, axis=0)
    tu = np.argmax(u_input, axis=0)
    tuw = np.argmax(u_w2, axis=0)

    # Compute avrage timing difference
    mask = (~(v2 < threshold).all(axis=0) * ~(u_input < threshold).all(axis=0))
    timing_before = np.mean(np.abs(tu[mask] - tv[mask]))
    timing_after = np.mean(np.abs(tuw[mask] - tv[mask]))

    # Print results
    print("\nAverage timing error for threshold={}mm/h (sample number = {})".format(threshold,np.sum(mask)))
    print("Before: {:.2f} h".format(timing_before))
    print("After:  {:.2f} h".format(timing_after))


#================================================================
# Plotting

if plot:
    print("\nPlotting")

    # Coordinates
    x2 = lat_u[14:-14]
    y2 = lon_u[14:-14]
    yy, xx = np.meshgrid(y2, x2, indexing='ij')
    yc = np.linspace(0, ny - 1, (2 ** I + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** I + 1), dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')

    # Define colormap
    nws_precip_colors = [
        'white',
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6",  # 8.00 - 10.00 inches
        "#fdfdfd"  # 10.00+
    ]
    precip_colormap = cls.ListedColormap(nws_precip_colors)
    clevs = [0, 0.02, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]  # , 150, 200, 250, 300, 400, 500, 600, 750]
    norm = cls.BoundaryNorm(clevs, 13)
    my_cmap = precip_colormap(np.arange(precip_colormap.N))
    my_cmap = ListedColormap(my_cmap)

    # Plot input fields u and v (background and contour respectively)
    lon_u2 = lon_u[14:-14]
    lat_u2 = lat_u[14:-14]
    lon_m = np.asarray(list(lon_u2) + [lon_u2[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u2) + [lat_u2[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(4, 6, figsize=(30, 15))
    pos = 0
    for kt in range(0, nt - 1):
        if kt == 12:
            av = axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_input[kt, :, :], cmap=my_cmap, norm=norm)
        else:
            axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_input[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].contour(yy, xx, v_input[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].scatter(lon_v, lat_v, c=v_station[kt, :], s=40, cmap=precip_colormap,
                                             edgecolor='black',
                                             norm=norm)
        axarr[int(pos / 6), pos % 6].set(adjustable='box-forced', aspect='equal')
        axarr[int(pos / 6), pos % 6].set_title("Time {}".format(kt))
        pos += 1
    plt.tight_layout()
    cbar = fig.colorbar(av, ax=axarr)
    cbar.set_label('mm/h')
    plt.savefig(folder_result + '/input_fields.png', dpi=200)
    plt.close()

    # Plot warped and "true" fields (background and contour respectively)
    lon_u2 = lon_u[14:-14]
    lat_u2 = lat_u[14:-14]
    u_w2 = u_warped[:, 14:-14, 14:-14]
    lon_m = np.asarray(list(lon_u2) + [lon_u2[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u2) + [lat_u2[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(4, 6, figsize=(30, 15))
    pos = 0
    for kt in range(0, nt - 1):
        if kt == 12:
            av = axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_w2[kt, :, :], cmap=my_cmap, norm=norm)
        else:
            axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_w2[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].contour(yy, xx, v2[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].scatter(lon_v, lat_v, c=v_station[kt, :], s=40, cmap=precip_colormap,
                                             edgecolor='black',norm=norm)
        axarr[int(pos / 6), pos % 6].set(adjustable='box-forced', aspect='equal')
        axarr[int(pos / 6), pos % 6].set_title("Time {}".format(kt))
        pos += 1
    plt.tight_layout()
    cbar = fig.colorbar(av, ax=axarr)
    cbar.set_label('mm/h')
    plt.savefig(folder_result + '/warped_field.png', dpi=200)
    plt.close()


    # Plot warped field and mapping (background and quiver respectively)
    fig, axarr = plt.subplots(4, 6, figsize=(30, 15))
    pos = 0
    for kt in range(0, nt - 1):
        if kt == 12:
            av = axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_w2[kt, :, :], cmap=my_cmap, norm=norm)
        else:
            axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_w2[kt, :, :], cmap=my_cmap, norm=norm)
        if reg == "a2":
            axarr[int(pos / 6), pos % 6].quiver(Tx, Ty, xxc - Tx, yyc - Ty,units='xy', scale=1)
        else:
            axarr[int(pos / 6), pos % 6].quiver(Tx[:, :, kt], Ty[:, :, kt], xxc - Tx[:, :, kt], yyc - Ty[:, :, kt],
                                            units='xy', scale=1)
        axarr[int(pos / 6), pos % 6].set(adjustable='box-forced', aspect='equal')
        axarr[int(pos / 6), pos % 6].set_title("Time {}".format(kt))
        pos += 1
    plt.tight_layout()
    cbar = fig.colorbar(av, ax=axarr)
    cbar.set_label('mm/h')
    plt.savefig(folder_result + '/warped_and_mapping.png', dpi=200)
    plt.close()

    #Plot mapping (grid)
    if reg == "a2":
        fig = plt.figure(figsize=(10, 10))
        plt.quiver(Tx, Ty, xxc - Tx, yyc - Ty, units='xy', scale=1)
        plt.axes().set_aspect("equal")
    else:
        fig, axarr = plt.subplots(3, 4, figsize=(17, 13), sharex=True, sharey=True)
        pos = 0
        for kt in range(8, 20):
            axarr[int(pos / 4), pos % 4].quiver(Tx[:, :, kt], Ty[:, :, kt], xxc - Tx[:, :, kt], yyc - Ty[:, :, kt],
                                                units='xy', scale=1,  width=0.04)
            axarr[int(pos / 4), pos % 4].set(adjustable='box-forced', aspect='equal')
            axarr[int(pos / 4), pos % 4].set_title("Time {}".format(kt))
            pos += 1
    plt.tight_layout()
    plt.savefig(folder_result + '/mapping.png', dpi=200)
    plt.close()



