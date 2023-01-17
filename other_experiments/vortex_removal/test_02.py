# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

from ast import literal_eval
import matplotlib.pyplot as plt
import os
import pandas as pd
import tc_formation.plots.observations as plt_obs
import xarray as xr

# # Checking Vortex Removal Scheme
#
# This notebook is just to show the result
# between removed and unremoved TC to see how
# the environmental field changes after using
# vortex removal algorithm by Kurihara 1993.

# +
removed_tc_label = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/tc_ibtracs_12h_WP_EP_v4.csv'
original_data_dir = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/'

label = pd.read_csv(removed_tc_label)
label['Other TC Locations'] = label['Other TC Locations'].apply(literal_eval)
label['Original Path'] = label['Path'].apply(lambda path: os.path.join(original_data_dir, os.path.basename(path)))
# -

# ## Checking non-TC files
#
# Non-TC files should be equal to each other.

# +
no_other_tc_mask = label['Other TC Locations'].apply(lambda locations: len(locations) == 0)
no_other_tc_df = label[no_other_tc_mask]
print('Number of no TC rows', len(no_other_tc_df))

for _, row in no_other_tc_df.sample(10).iterrows():
    original_data = xr.open_dataset(row['Original Path'], engine='netcdf4')
    removed_data = xr.open_dataset(row['Path'], engine='netcdf4')

    print((original_data == removed_data).all())
# -

# ## Checking TC files
#
# Dataset with TC should show no tropical after removal.

# +
other_tc_df = label[~no_other_tc_mask]
print('Number of TC rows', len(other_tc_df))

for _, row in other_tc_df.sample(10).iterrows():
    original_data = xr.open_dataset(row['Original Path'], engine='netcdf4')
    removed_data = xr.open_dataset(row['Path'], engine='netcdf4')
    
    # Make sure that they're different.
    print((original_data == removed_data).all())

    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))

    ax = axes[0]
    ax.set_title('Original Data')
    plt_obs.plot_variablef(dataset=original_data, variable='absvprs', pressure_level=700, ax=ax)
    plt_obs.plot_wind(dataset=original_data, pressure_level=850, ax=ax, skip=4)

    ax = axes[1]
    ax.set_title('Removed Data')
    plt_obs.plot_variablef(dataset=removed_data, variable='absvprs', pressure_level=700, ax=ax)
    plt_obs.plot_wind(dataset=removed_data, pressure_level=850, ax=ax, skip=4)

    print('\n====', row['Path'], row['Original Path'])
    print(row['Other TC Locations'])
    display(fig)
    plt.close(fig)
