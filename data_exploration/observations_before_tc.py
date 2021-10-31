# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os  # noqa
os.environ['PROJ_LIB'] = '.conda/envs/tc_prediction/share/proj'  # noqa

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr
import pandas as pd
# -

# Load all the TC to see what we got.
data_home = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb/'
tc = pd.read_csv(os.path.join(data_home, 'tc.csv'), dtype={
    'Observation': 'str',
    'Genesis': 'str',
    'End': 'str'
})
tc[tc['TC'] == 1]

# Of these storms, how many of them have positive latitude
tc[(tc['TC'] == 1) & (tc['Latitude'] > 0)]

# Of these storms, how many of them have latitude from 5 to 45
tc[(tc['TC'] == 1) & (tc['Latitude'] <= -5) & (tc['Latitude'] >= -45)]

# Of these storms, how many of them have latitude outside 5 to 45
tc[(tc['TC'] == 1) & ~((tc['Latitude'] <= -5) & (tc['Latitude'] >= -45))]

# +
ds = xr.open_dataset(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb/fnl_20080501_00_00.nc')
lat = ds['lat']
lon = ds['lon']

ds
# -

nbuf = 7
buff = nbuf*0.03

# +
fig, ax = plt.subplots(figsize=(20, 10))
basemap = Basemap(
    projection='cyl',
    llcrnrlon=np.nanmin(lon + buff),
    llcrnrlat=np.nanmin(lat + buff),
    urcrnrlon=np.nanmax(lon - buff),
    urcrnrlat=np.nanmax(lat - buff),
    resolution='h')

parallels = np.arange(-90, 90, 5.)
meridians = np.arange(-180, 180, 5.)
basemap.drawparallels(
    parallels, labels=[1, 0, 0, 0], fontsize=18, color="grey")
basemap.drawmeridians(
    meridians, labels=[0, 0, 0, 1], fontsize=18, color="grey", rotation=45)

basemap.drawcoastlines()
basemap.drawstates()
basemap.drawcountries()
basemap.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
basemap.drawcounties()

x, y = np.meshgrid(lon, lat)
dclevs = np.arange(-1, 1, 0.2)
nx = 351
ny = 351
bigarray = ds['tmpsfc']
cs = basemap.contourf(x, y, bigarray, dclevs, cmap='bwr')
cs = basemap.contourf(x, y, bigarray, cmap='rainbow')
cb = basemap.colorbar(cs, "right", size="5%", pad="2%")
