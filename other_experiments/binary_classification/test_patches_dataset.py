# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %cd ../..
from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from tc_formation.binary_classifications.data.patches_classification_data_loader import *
import xarray as xr
# -

# # Test Patches Dataset

# ## Test `extract_patches`

# +
def plot_patch(ds: xr.Dataset, title: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    lat, lon = ds['lat'].values, ds['lon'].values
    minlat, maxlat = lat.min(), lat.max()
    minlon, maxlon = lon.min(), lon.max()

    bm = Basemap(
        llcrnrlat=minlat,
        llcrnrlon=minlon,
        urcrnrlat=maxlat,
        urcrnrlon=maxlon,
        ax=ax)

    xx, yy = np.meshgrid(lon, lat)

    bm.drawcoastlines()
    bm.drawcountries()
    bm.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
    bm.drawmeridians(
        np.arange(minlon, maxlon, 20),
        labels=[0, 0, 0, 1],
        color="grey")
    bm.drawparallels(
        np.arange(minlat, maxlat, 5),
        labels=[1, 0, 0, 0],
        color="grey")

    cs = bm.contourf(x=xx, y=yy, data=ds['tmpsfc'], latlon=True)
    fig.colorbar(cs, ax=ax)
    ax.set_title(title)
    fig.tight_layout()


path = './data/theanh_WPAC_baseline_5/baseline_20000505_12_00.nc'
ds = xr.load_dataset(path, engine='netcdf4')

# First, we will plot the file.
plot_patch(ds)
# -

# Next, we will extract patches to see if they're good.
patches = extract_patches(ds, domain_size=30, stride=5)
for p, coord in patches:
    print(p['tmpsfc'].values.shape)
    plot_patch(p, title=f'{coord=}')

# ## Test `load_xr_dataset_as_patches`

# +
subset = OrderedDict(
    tmpsfc=True,
)
patches, coords, _ = load_xr_dataset_as_patches(
    path, subset=subset, domain_size=30, stride=5)
assert len(patches) == 10, 'Because there are 10 patches above!'
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for patch, coord, ax in zip(patches, coords, axes.flatten()):
    ax.pcolormesh(patch[:, :, 0])
    ax.set_title(f'lat={coord[0]:.2f}, lon={coord[1]:.2f}')

fig.tight_layout()
# -

# ## Test `PatchesClassificationDataLoader`

dataloader = PatchesClassificationDataLoader(
    domain_size=30, stride=5, subset=subset, keep_hours=[0])
ds = dataloader.load_dataset_without_label('./data/theanh_WPAC_baseline_5', batch_size=4)
for X, coords, _ in iter(ds):
    print(f'{coords=}')
    print(f'{X.shape=}')
    break
