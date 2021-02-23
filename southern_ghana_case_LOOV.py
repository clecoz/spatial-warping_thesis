from registration_approach3 import *
from registration_approach2 import *
from registration_approach1 import *
from pykrige import OrdinaryKriging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import combinations


#=======================================================================================================================
# Southern Ghana case - "LOOV" experiment
#=======================================================================================================================

# Choose case
case = '20180422'
folder = case

# Choose folder where to save results (and create it if does not exist)
folder_result = "{}/results_LOOV/".format(case).replace('.','p')
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

# Choose if you want to plot or compute statistics
plot = True
stats = True
threshold = 0.1   # in mm/h, used to compute the position and timing error

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


u_input = u.copy()

#==================================================================
# Pre-processing of u
print('\nPre-processing of u')

# Extend the domain
x = np.array(np.arange(-4.75, 1.75, 0.1))
y = np.array(np.arange(3.25, 9.75, 0.1))
xx, yy = np.meshgrid(x, y, indexing='ij')
ny = len(y)
nx = len(x)

u = np.zeros((nt, nx, ny))
u[:, 14:-14, 14:-14] = u_input

u_o = u.copy()

# Remove light precipitation (<1mm)
u[u < 1] = 0


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
# Start LOOV
# We are looping on the stations, removing one from the input before the interpolation to obtain the field v.
# This station can then be used later for validation.

# Create array where we will save the maapings for each iteration
if reg == "a1" or reg == "a3":
    Tx_loov = np.zeros((2 ** I + 1, 2 ** I + 1, nt, ns))
    Ty_loov = np.zeros((2 ** I + 1, 2 ** I + 1, nt, ns))
elif reg == "a2":
    Tx_loov = np.zeros((2 ** I + 1, 2 ** I + 1, ns))
    Ty_loov = np.zeros((2 ** I + 1, 2 ** I + 1, ns))
else:
    print("Error (wrong ""reg""): this approach does not exist.")


# Start iteration
for ks in range(ns):
    print('\n- Remove station {} ({}, max:{})'.format(ks,station_ID[ks],np.max(v_station[:,ks])))
    # Remove one station
    v_station_LOOV = np.delete(v_station,ks,axis=1)
    lon_LOOV = np.delete(lon_v,ks,axis=0)
    lat_LOOV = np.delete(lat_v, ks, axis=0)

    # ---------------------------------------------
    # TAHMO interpolation
    v_input = np.zeros((nt,37,37))
    for k in range(nt):
        OK = OrdinaryKriging(lon_LOOV, lat_LOOV, np.sqrt(v_station_LOOV[k, :]), variogram_model='exponential',
                             variogram_parameters={'sill': 1.0, 'range': 2, 'nugget': 0.01}, nlags=50, verbose=False,
                             enable_plotting=False, weight=True, coordinates_type='geographic')
        z, ss = OK.execute('grid', lon_u, lat_u)
        v_input[k, :, :] = (z ** 2).T


    # ---------------------------------------------
    # Pre-procesing of v
    # Extend the domain
    v = np.zeros((nt, nx, ny))
    v[:, 14:-14, 14:-14] = v_input

    # Remove light precipitation (<1mm)
    v[v < 1] = 0

    # ---------------------------------------------
    # Define mask
    # (the mask is based on the kriging variance, and so depend on the station locations)
    mask = np.zeros((nt, nx, ny))
    for kt in range(nt):
        mask[kt, 14:-14, 14:-14] = (ss < 0.5).astype("float").T


    # ---------------------------------------------
    # Registration
    if reg == "a1":
        Tx, Ty = registration_a1(u, v, x, y, I, c1, c2, c3, mask, folder_result, ks)
        Tx_loov[:, :, :, ks] = Tx
        Ty_loov[:, :, :, ks] = Ty
    elif reg == "a2":
        Tx, Ty = registration_a2(u, v, x, y, I, c1, c2, c3, mask, folder_result, ks)
        Tx_loov[:, :, ks] = Tx
        Ty_loov[:, :, ks] = Ty
    elif reg == "a3":
        Tx, Ty = registration_a3(u, v, x, y, I, c1, c2, c3, c4, mask, Acomb, time_cor, folder_result, ks)
        Tx_loov[:, :, :, ks] = Tx
        Ty_loov[:, :, :, ks] = Ty
    else:
        print("Error (wrong ""reg""): this approach does not exist.")




#================================================================
# Warping
print("\nWarping")

# Warped field at the station locations for the ns iteration
u_warped_loov = np.zeros((nt,ns))
if reg == "a1" or reg == "a3":
    for ks in range(ns):
        u_warped_loov[:, ks] = np.squeeze(
            mapped_TAHMO(u_o, y, x, Ty_loov[:, :, :, ks], Tx_loov[:, :, :, ks], lat_v, lon_v, I))[:, ks]
elif reg == "a2":
    for ks in range(ns):
        u_warped_loov[:, ks] = np.squeeze(
            mapped_TAHMO(u_o, y, x, Ty_loov[:, :, ks], Tx_loov[:, :, ks], lat_v, lon_v, I))[:, ks]
else:
    print("Error (wrong ""reg""): this approach does not exist.")


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
    RMSE_after = np.sqrt(np.mean((u_warped_loov - v_station) ** 2))
    MAE_after = np.mean(np.abs(u_warped_loov - v_station))
    C_after = np.corrcoef(u_warped_loov.reshape(-1), v_station.reshape(-1))[0, 1]

    print('#---------------------------------------#')
    print('#        |  RMSE  |  MAE  | Correlation #')
    print('#---------------------------------------#')
    print('# Before |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_before,MAE_before,C_before))
    print('# After  |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_after,MAE_after,C_after))
    print('#---------------------------------------#')


    # ---------------------------------------------
    # Timing error
    # The timming errors are computed at the station locations where we have gauge measurements

    # Find indexes of the pixel with the max. rain
    tv = np.argmax(v_station, axis=0)
    tu = np.argmax(u_station, axis=0)
    tuw = np.argmax(u_warped_loov, axis=0)

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


    # Plot average mapping (over LOOV iterations)
    if reg == "a2":
        fig = plt.figure(figsize=(8,8))
        mean_Tx = np.mean(Tx_loov, axis=2)
        mean_Ty = np.mean(Ty_loov, axis=2)
        max_Tx = np.max(Tx_loov, axis=2) - mean_Tx
        max_Ty = np.max(Ty_loov, axis=2) - mean_Ty
        min_Tx = np.min(Tx_loov, axis=2) - mean_Tx
        min_Ty = np.min(Ty_loov, axis=2) - mean_Ty
        yerr = np.zeros((2, len(mean_Ty.reshape(-1))))
        yerr[0, :] = -min_Ty.reshape(-1)
        yerr[1, :] = max_Ty.reshape(-1)
        xerr = np.zeros((2, len(mean_Ty.reshape(-1))))
        xerr[0, :] = -min_Tx.reshape(-1)
        xerr[1, :] = max_Tx.reshape(-1)
        plt.quiver(mean_Tx, mean_Ty, xxc - mean_Tx, yyc - mean_Ty, units='xy', scale=1)
        plt.errorbar(mean_Tx.reshape(-1), mean_Ty.reshape(-1), yerr=yerr, xerr=xerr,ecolor='r', fmt='.k', ms=1)
    else:
        fig, axarr = plt.subplots(2, 4, figsize=(20, 10))
        pos = 0
        for kt in range(0, 8):
            j = kt + 9
            mean_Tx = np.mean(Tx_loov[:, :, j, :], axis=2)
            mean_Ty = np.mean(Ty_loov[:, :, j, :], axis=2)
            max_Tx = np.max(Tx_loov[:, :, j, :], axis=2) - mean_Tx
            max_Ty = np.max(Ty_loov[:, :, j, :], axis=2) - mean_Ty
            min_Tx = np.min(Tx_loov[:, :, j, :], axis=2) - mean_Tx
            min_Ty = np.min(Ty_loov[:, :, j, :], axis=2) - mean_Ty
            yerr = np.zeros((2, len(mean_Ty.reshape(-1))))
            yerr[0, :] = -min_Ty.reshape(-1)
            yerr[1, :] = max_Ty.reshape(-1)
            xerr = np.zeros((2, len(mean_Ty.reshape(-1))))
            xerr[0, :] = -min_Tx.reshape(-1)
            xerr[1, :] = max_Tx.reshape(-1)
            axarr[int(pos / 4), pos % 4].quiver(mean_Tx, mean_Ty, xxc - mean_Tx, yyc - mean_Ty, units='xy', scale=1)
            axarr[int(pos / 4), pos % 4].errorbar(mean_Tx.reshape(-1), mean_Ty.reshape(-1), yerr=yerr, xerr=xerr,
                                                  ecolor='r', fmt='.k', ms=1)
            axarr[int(pos / 4), pos % 4].set_title(datetime[j])
            pos += 1
    plt.savefig(folder_result + 'average_mapping.png', dpi=200)
    plt.close()

