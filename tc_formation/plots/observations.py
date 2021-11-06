from . import decorators as _d
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr


@_d._with_axes
@_d._with_basemap
def plot_wind(dataset: xr.Dataset, pressure_level: int, basemap: Basemap = None, ax: plt.Axes = None):
    lats, longs = np.meshgrid(dataset['lat'], dataset['lon'])
    u_wind = dataset['ugrdprs'].sel(lev=pressure_level)
    v_wind = dataset['vgrdprs'].sel(lev=pressure_level)
    basemap.barbs(lats, longs, u_wind, v_wind)


@_d._with_axes
@_d._with_basemap
def plot_variable(dataset: xr.Dataset, variable: str, pressure_level: int, cmap='rainbow', basemap: Basemap = None, ax: plt.Axes = None):
    lats, longs = np.meshgrid(dataset['lat'], dataset['lon'])
    data = dataset[variable].sel(lev=pressure_level)
    basemap.contourf(lats, longs, data, cmap=cmap)
