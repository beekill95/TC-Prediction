# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt
import pandas as pd
from tc_formation.plots import observations as plt_obs
import numpy as np
import xarray as xr

# # Daily Genesis Potential Test 02
#
# What is this test about?
#
# Plotting individual parameters in the genesis potential index.

label_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
label_df = pd.read_csv(label_path)
label_with_tc_df = label_df[label_df['TC']]
label_without_tc_df = label_df[~label_df['TC']]

# # Single Observations

# +
from ast import literal_eval # noqa
from matplotlib.patches import Rectangle # noqa
import tc_formation.genesis_potential.genesis_potential_index as gpi # noqa


def plot_rectangle(center, ax, color='blue', size=5):
    half_size = size / 2.0
    center = np.asarray(center)
    rec = Rectangle(center - half_size, size, size, color=color, fill=False, lw=2.)
    ax.add_patch(rec)


def plot_param(row, param_fn, use_log=False, ax=None):
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    param_values = param_fn(ds)

    print(row['Latitude'], row['Longitude'])
    other_tc_locations = literal_eval(row['Other TC Locations'])

    # Plot GPI values.
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    fig = ax.get_figure()
    yy, xx = np.meshgrid(ds.lon, ds.lat)
    if use_log:
        cs = ax.contourf(yy, xx, np.log(param_values + 1e-6))
    else:
        cs = ax.contourf(yy, xx, param_values)
    fig.colorbar(cs, ax=ax)

    # Not good, but usable.
    ax.set_ylim(5, 45)
    ax.set_xlim(100, 260)
    ax.set_title(f'{param_fn.__name__} for {row["Date"]}')

    # Plot rectangles indicating location of tropical cyclogenesis
    # as well as other tropical cyclones.
    plot_rectangle((row['Longitude'], row['Latitude']), ax=ax)
    for loc in other_tc_locations:
        plot_rectangle(loc[::-1], ax=ax, color='orange')

    fig.tight_layout()
    return fig

# + [markdown] tags=[]
# ## Vertical Wind Shear

# + tags=[]
for _, row in label_with_tc_df.sample(10).iterrows():
    fig = plot_param(row, gpi.vertical_shear_parameter, use_log=False)
    display(fig)
    plt.close(fig)

# + tags=[]
for _, row in label_without_tc_df.sample(10).iterrows():
    fig = plot_param(row, gpi.vertical_shear_parameter, use_log=False)
    display(fig)
    plt.close(fig)
# -

# ## Absolute Vorticity

for _, row in label_with_tc_df.sample(10).iterrows():
    fig = plot_param(row, gpi.vorticity_parameter, use_log=False)
    display(fig)
    plt.close(fig)

# + tags=[]
for _, row in label_without_tc_df.sample(10).iterrows():
    fig = plot_param(row, gpi.vorticity_parameter, use_log=False)
    display(fig)
    plt.close(fig)
