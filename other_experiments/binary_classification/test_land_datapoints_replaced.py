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

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
from typing import Union
import xarray as xr
# -

original_dir = '/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_WP_tc_binary'
replaced_dir = 'data/ncep_WP_land_replaced_tc_binary'
file_path = 'pos/20080506_18_00_7.2_134.2_2008128N07134.nc'

original_ds = xr.load_dataset(os.path.join(original_dir, file_path))
original_ds

replaced_ds = xr.load_dataset(os.path.join(replaced_dir, file_path))
replaced_ds

# +
def plot_comparison(
        original_ds: xr.Dataset,
        replaced_ds: xr.Dataset,
        variables: list[tuple[str, Union[int, None]]]):
    def get_value(ds, varname, lev):
        var = ds[varname]
        if lev is not None:
            var = var.sel(lev=lev)

        return var.values

    fig, axes = plt.subplots(nrows=len(variables), ncols=2, figsize=(8, 12))

    for row, (varname, lev) in enumerate(variables):
        original_val = get_value(original_ds, varname, lev)
        replaced_val = get_value(replaced_ds, varname, lev)

        ax = axes[row, 0]
        xx, yy = np.meshgrid(original_ds['lon'].values, original_ds['lat'].values)
        m = Basemap(
            llcrnrlon=original_ds['lon'].min(),
            llcrnrlat=original_ds['lat'].min(),
            urcrnrlon=original_ds['lon'].max(),
            urcrnrlat=original_ds['lat'].max(),
            resolution='i',
            projection='cyl',
            ax=ax,
        )
        xx, yy = m(xx, yy)
        m.drawcoastlines()
        cs = m.pcolormesh(xx, yy, original_val)
        fig.colorbar(cs, ax=ax)
        ax.set_title(f'Original {varname} at {lev=}')

        ax = axes[row, 1]
        xx, yy = np.meshgrid(replaced_ds['lon'].values, replaced_ds['lat'].values)
        m = Basemap(
            llcrnrlon=replaced_ds['lon'].min(),
            llcrnrlat=replaced_ds['lat'].min(),
            urcrnrlon=replaced_ds['lon'].max(),
            urcrnrlat=replaced_ds['lat'].max(),
            resolution='i',
            projection='cyl',
            ax=ax,
        )
        m.drawcoastlines()
        xx, yy = m(xx, yy)
        cs = m.pcolormesh(xx, yy, replaced_val, vmin=original_val.min(), vmax=original_val.max())
        fig.colorbar(cs, ax=ax)
        ax.set_title(f'Replaced {varname} at {lev=}')

    fig.tight_layout()

vars_to_compare = [
    ('absvprs', 1000),
    ('absvprs', 500),
    ('pressfc', None),
    ('tmpsfc', None),
]
plot_comparison(original_ds, replaced_ds, vars_to_compare)
