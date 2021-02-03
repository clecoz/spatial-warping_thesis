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

#=======================================================================================================================
# Southern Ghana case - "All" experiment
#=======================================================================================================================

# Choose case
case = '20180422'
folder = case

# Choose folder where to save results (and create it if does not exist)
folder_result = "{}/results/".format(case).replace('.','p')
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

# Choose if you want to plot or compute statistics
plot = True
stats = True
threshold = 10   # in mm/h, used to compute the position and timing error

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
TAHMO_file = './{}/TAHMO_{}.csv'.format(folder,case)
IMERG_file = './{}/IMERG_all_{}.csv'.format(folder,case)
file_coord_sta = './{}/coord_{}.csv'.format(folder,case)
file_coord_IMERG = './{}/coord_IMERG.csv'.format(folder)

# Read data from files
df_TAHMO = pd.read_csv(TAHMO_file, index_col=0)
df_IMERG = pd.read_csv(IMERG_file, index_col=0)
df_coord_sta = pd.read_csv(file_coord_sta, index_col=0)
df_coord_IMERG = pd.read_csv(file_coord_IMERG, index_col=0)

# Data to be corrected (IMERG-Late)
u = df_IMERG.values.reshape((-1,37,37))
lon_u = df_coord_IMERG.loc['lon'].values
lat_u = df_coord_IMERG.loc['lat'].values
npts = lon_u.shape[0]

# Reference data (TAHMO)
lon_v = df_coord_sta.loc['lon'].values
lat_v = df_coord_sta.loc['lat'].values
v_station = df_TAHMO.values
station_ID = df_TAHMO.columns.values

# Coordinates
nt, ns = v_station.shape
t = np.array(range(0, nt))
datetime = df_TAHMO.index.values
x = lon_u
y = lat_u
nx = len(x)
ny = len(y)

# Interpolation of (TAHMO) stations on the regular grid
v = np.zeros((nt,37,37))
for k in range(nt):
    OK = OrdinaryKriging(lon_v, lat_v, np.sqrt(v_station[k, :]), variogram_model='exponential',
                         variogram_parameters={'sill': 1.0, 'range': 2, 'nugget': 0.01}, nlags=50, verbose=False,
                         enable_plotting=False, weight=True, coordinates_type='geographic')
    z, ss = OK.execute('grid', lon_u, lat_u)
    v[k, :, :] = (z ** 2).T


u_input = u.copy()
v_input = v.copy()

#==================================================================
# Pre-processing
print('\nPre-processing')

# Extend the domain
x = np.array(np.arange(-4.75, 1.75, 0.1))
y = np.array(np.arange(3.25, 9.75, 0.1))
xx, yy = np.meshgrid(x, y, indexing='ij')
ny = len(y)
nx = len(x)

u = np.zeros((nt, nx, ny))
u[:, 14:-14, 14:-14] = u_input
v = np.zeros((nt, nx, ny))
v[:, 14:-14, 14:-14] = v_input

u_o = u.copy()
v_o = v.copy()
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
for kt in range(nt):
    mask[kt, 14:-14, 14:-14] = (ss < 0.5).astype("float").T



#================================================================
# Registration
print("\nStart regisration")

ks = 'all'          # this parameter is used in the LOOV experiment

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

u_warped = mapped(u_o, y, x, Ty, Tx, I)                           # Warped field at the station locations
u_warped_station = mapped_TAHMO(u_o,y,x,Ty,Tx,lat_v,lon_v,I)     # Warped field at the grid points


#================================================================
# Statistics

if stats:
    print('\nStatistics')
    # The statistics are computed at the station locations where we have gauge measurements

    # Values of u (IMERG) at the station locations
    # (already prepared and saved, we only need to read the file)
    IMERG_station_file = './{}/IMERG_{}.csv'.format(folder, case)
    df_IMERG_sta = pd.read_csv(IMERG_station_file, index_col=0)
    u_station = df_IMERG_sta.values


    # Statistics before warping
    RMSE_before = np.sqrt(np.mean((u_station - v_station) ** 2))
    MAE_before = np.mean(np.abs(u_station - v_station))
    C_before = np.corrcoef(u_station.reshape(-1), v_station.reshape(-1))[0, 1]

    # Statistics after warping
    RMSE_after = np.sqrt(np.mean((u_warped_station - v_station) ** 2))
    MAE_after = np.mean(np.abs(u_warped_station - v_station))
    C_after = np.corrcoef(u_warped_station.reshape(-1), v_station.reshape(-1))[0, 1]

    print('#---------------------------------------#')
    print('#        |  RMSE  |  MAE  | Correlation #')
    print('#---------------------------------------#')
    print('# Before |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_before,MAE_before,C_before))
    print('# After  |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_after,MAE_after,C_after))
    print('#---------------------------------------#')


    # ---------------------------------------------
    # Position error
    # For the position error, we use the field v (interpolated stations) as truth because we need a 2D field to compute position and distance

    # We use the original domain not the extended one
    u_w2 = u_warped[:, 14:-14, 14:-14]
    x2 = lon_u
    y2 = lat_u

    # Find indexes of the pixel with the max. rain
    iu, ju = np.unravel_index(np.argmax(u_input.reshape(nt, -1), axis=1), (37, 37))
    iv, jv = np.unravel_index(np.argmax(v_input.reshape(nt, -1), axis=1), (37, 37))
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
    mask = (~(v_input<threshold).all(axis=(1,2)) * ~(u_input<0.1).all(axis=(1,2)))
    average_dist_before = np.mean(dist_before[mask])
    average_dist_after = np.mean(dist_after[mask])

    # Print results
    print("\nAverage position error for threshold={}mm/h (sample number = {})".format(threshold, np.sum(mask)))
    print("Before: {:.2f} km".format(average_dist_before))
    print("After:  {:.2f} km".format(average_dist_after))


    # ---------------------------------------------
    # Timing error
    # The timming errors are computed at the station locations where we have gauge measurements

    # Find indexes of the pixel with the max. rain
    tv = np.argmax(v_station, axis=0)
    tu = np.argmax(u_station, axis=0)
    tuw = np.argmax(u_warped_station, axis=0)

    # Compute avrage timing difference
    mask = (~(v_station < threshold).all(axis=0) * ~(u_station < 0.1).all(axis=0))
    timing_before = np.mean(np.abs(tu[mask] - tv[mask]))
    timing_after = np.mean(np.abs(tuw[mask] - tv[mask]))

    # Print results
    print("\nAverage timing error for threshold={}mm/h (sample number = {})".format(threshold, np.sum(mask)))
    print("Before: {:.2f} h".format(timing_before))
    print("After:  {:.2f} h".format(timing_after))


#================================================================
# Plotting

if plot:
    print("\nPlotting")

    # Coordinates
    x2 = lat_u
    y2 = lon_u
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
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
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

    # Plot warped field and interpolated stations (background and contour respectively)
    lon_u2 = lon_u
    lat_u2 = lat_u
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
        axarr[int(pos / 6), pos % 6].contour(yy, xx, v_input[kt, :, :], cmap=my_cmap, norm=norm)
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





