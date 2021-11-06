from . import decorators as _d
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr


@_d._with_axes
@_d._with_basemap
def plot_wind(dataset: xr.Dataset, pressure_level, basemap: Basemap, ax: plt.Axes):
    latitude = dataset['lat']
    longitude = dataset['lon']

    lats, longs = np.meshgrid(longitude, latitude)
    u_wind = dataset['ugrdprs'].sel(lev=pressure_level)
    v_wind = dataset['vgrdprs'].sel(lev=pressure_level)

    basemap.barbs(lats, longs, u_wind, v_wind)
