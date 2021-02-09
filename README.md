# spatial-warping_thesis

This project permits to reproduce the results presented in Chapter 3 of the PhD thesis "Straight as rain, Africa's rainfall measured, modeled, mapped and analyzed" (available at https://repository.tudelft.nl/). 
The goal is to gauge-adjust rainfall estimate with respect to the position of the rain events by using a spatial warping method.

## Requirement
The python scripts use the following libraries:
 - os
 - time
 - numpy
 - scipy
 - scipy.sparse
 - scipy.optimize
 - itertools
 - pandas
 - math
 - PyKrige
 - matplotlib.colors
 - matplotlib.pyplot
 - mpl_toolkits.basemap

## How to use
There are three main scripts:

- "synthetic_case.py" permits to reproduce the synthetic case presented in Section 3.3. One can choose the experiment ("Full" or "Interpolated"), the registration approach and parameters. The script reads the input data from the "case" folder and then applies the automatic registration and the warping methods. It is possible to produce some statistics and plots by setting the variables "stats" and "plot" to "True".
  
 - "southern_ghana_case.py" permits to reproduce the southern Ghana case presented in Section 3.4 for the "All" experiment. One can select the registration approach and change the parameters. The script reads the input data from the "case" folder. The input data consists in gauge measurements from the TAHMO (Trans-African Hydro-Meteorological Network) and rainfall estimates from IMERG-Late. The automatic registration and the warping methods are then applied to the data. It is possible to produce some statistics and plots by setting the variables "stats" and "plot" to "True".
 
 - "southern_ghana_case_LOOV.py" permits to reproduce the LOOV experiment for the southern Ghana case. One can modify the approach and the parameters of the registration.
 
 These scripts can be applied to ohter cases, given that the input data have the same format as the one provided in the "case" folders.
