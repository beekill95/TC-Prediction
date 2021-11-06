from . import decorators as _d
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from typing import Union
import xarray as xr


@_d._with_axes
@_d._with_basemap
def plot_wind(dataset: xr.Dataset, pressure_level: Union[int, dict], basemap: Basemap = None, *args, **kwargs):
    if isinstance(pressure_level, int):
        pressure_level = dict(lev=pressure_level)

    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    u_wind = dataset['ugrdprs'].sel(**pressure_level)
    v_wind = dataset['vgrdprs'].sel(**pressure_level)
    basemap.barbs(lats, longs, u_wind, v_wind)


@_d._with_axes
@_d._with_basemap
def plot_variable(
        dataset: xr.Dataset,
        variable: str,
        pressure_level: Union[int, dict] = None,
        cmap='rainbow',
        basemap: Basemap = None,
        *args, **kwargs):
    if isinstance(pressure_level, int):
        pressure_level = dict(lev=pressure_level)

    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    data = dataset[variable]
    data = data if pressure_level is None else data.sel(**pressure_level)

    cs = basemap.contourf(lats, longs, data, cmap=cmap)
    basemap.colorbar(cs, "right", size="5%", pad="2%")
