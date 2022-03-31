import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(30, 8))

    basemap = Basemap(
        projection='cyl',
        llcrnrlon=100,
        llcrnrlat=5,
        urcrnrlon=260,
        urcrnrlat=45,
        resolution='h',
        ax=ax)

    parallels = np.arange(-90, 90, 5.)
    meridians = np.arange(-180, 180, 20)
    basemap.drawparallels(
        parallels,
        labels=[1, 0, 0, 0],
        fontsize=18,
        color="grey")
    basemap.drawmeridians(
        meridians,
        labels=[0, 0, 0, 1],
        fontsize=18,
        color="grey")

    basemap.drawcoastlines()
    basemap.drawstates()
    basemap.drawcountries()
    basemap.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
    basemap.drawcounties()

    fig.tight_layout()
    fig.savefig('map.jpg')
