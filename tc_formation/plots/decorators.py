from functools import wraps
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


def _with_basemap(func):
    @wraps(func)
    def function_with_basemap(dataset=None, basemap=None, ax=None, *args, **kwargs):
        assert ax is not None, 'Required `ax`'
        latitude = dataset['lat']
        longitude = dataset['lon']

        if basemap is None:
            try:
                basemap = ax.__d_basemap
            except AttributeError:
                basemap = Basemap(
                    projection='cyl',
                    llcrnrlon=np.nanmin(longitude),
                    llcrnrlat=np.nanmin(latitude),
                    urcrnrlon=np.nanmax(longitude),
                    urcrnrlat=np.nanmax(latitude),
                    resolution='h',
                    ax=ax)

                parallels = np.arange(-90, 90, 5.)
                meridians = np.arange(-180, 180, 20)
                basemap.drawparallels(
                    parallels,
                    labels=[1, 0, 0, 0],
                    fontsize=30,
                    color="grey")
                basemap.drawmeridians(
                    meridians,
                    labels=[0, 0, 0, 1],
                    fontsize=30,
                    color="grey")

                basemap.drawcoastlines()
                basemap.drawstates()
                basemap.drawcountries()
                basemap.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
                basemap.drawcounties()

                # Assign basemap to ax for future reference.
                ax.__d_basemap = basemap

        return func(*args, dataset=dataset, basemap=basemap, ax=ax, **kwargs)

    return function_with_basemap


def _with_axes(func):
    @wraps(func)
    def function_with_axes(ax=None, figsize=(10, 8), *args, **kwargs):
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        func(*args, ax=ax, **kwargs)
        return ax.get_figure(), ax

    return function_with_axes
