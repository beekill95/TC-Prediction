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
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb/fnl_20210809_12_00.nc')
lat = ds['lat']
lon = ds['lon']

ds
# -

nbuf = 7
buff = 0

# +
import numpy as np

fig, ax = plt.subplots(figsize=(30, 15))
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
bigarray = ds['tmpsfc']
wind_u = ds['ugrdprs'][1,:,:]
wind_v = ds['vgrdprs'][1,:,:]
#cs = basemap.contourf(x, y, wind_v, cmap='rainbow')
cs = basemap.contourf(x, y, bigarray, cmap='rainbow')
#uv = basemap.barbs(x, y, wind_u, wind_v)

speed = np.sqrt(wind_u*wind_u + wind_v*wind_v)
#uv = basemap.streamplot(x, y, wind_u, wind_v, latlon=True)

u_mask=wind_u[(wind_u > 17) | (wind_u < -17)]
v_mask=wind_v[(wind_v > 17) | (wind_v < -17)]
uv = basemap.quiver(x, y, u_mask, v_mask)
#cb = basemap.colorbar(cs, "right", size="5%", pad="2%")
# -

np.nanmin(wind_u)
(wind_u > 17) | (wind_u < -17)

a = np.asarray([True, False])
b = np.asarray([False, False])
a | b
