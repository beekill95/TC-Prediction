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

# # Daily Genesis Potential Test 01
#
# What is this test about?
#
# TODO:

label_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
label_df = pd.read_csv(label_path)
label_with_tc_df = label_df[label_df['TC']]
label_without_tc_df = label_df[~label_df['TC']]

# ## Genesis Potential Index
#
# Based on the formula shown in [Gray](https://mountainscholar.org/bitstream/handle/10217/247/0234_Bluebook.pdf;sequence=1)
#
# (Genesis Potential) ∝ (Vorticity Parameter) (Corriolis Parameter) (Vertical Shear Parameter) (Ocean Thermal Energy) (Moist Stability Parameter) (Humidity Parameter)
#
# These parameters are further grouped into 2 categories:
#
# * Dynamic Potential: Vorticity parameter, Corriolis parameter, and Vertical Shear parameter.
# * Thermal Potential: Ocean Thermal energy, Moist Stability parameter, and Humidity parameter.
#
# Details of these parameters:
#
# * Vorticity parameter: $(\zeta_r + 5)$ at 950mb where $\zeta_r$ is in $10^{-6} \text{s}^{-1}$
# * Corriolis parameter: $2\Omega sin\phi$ where $\Omega$ is rotation rate of the Earth,
# and $\phi$ is the latitude.
# * Vertical Shear parameter: $1 / (S_z + 3)$ where $S_z = |\partial V / \partial p|$
# where $S_z$ is in m/s per 750mb.
# * Ocean Thermal energy: $\int_{60m}^{sfc} \rho_w c_w (T - 26)$
# where $\rho_w$ and $c_w$ are density and specific heat capacity of water respectively.
# E is in $10^3 cal/m^3$.
# * Moist Stability parameter: $\partial \theta_e / \partial p + 5$ is in K per 500mb.
# * Relative Humidity parameter: $\frac{\overline{RH} - 40}{30}$
# where $\overline{RH}$ is mean relative humidity between 700mb and 500mb.
# Parameter is 0 for $\overline{RH} < 40$ and 1 for $\overline{RH} \ge 70$.

# ### Single Observation

# +
from ast import literal_eval # noqa
from matplotlib.patches import Rectangle # noqa
import tc_formation.genesis_potential.genesis_potential_index as gpi # noqa


def plot_rectangle(center, ax, color='blue', size=5):
    half_size = size / 2.0
    center = np.asarray(center)
    rec = Rectangle(center - half_size, size, size, color=color, fill=False, lw=2.)
    ax.add_patch(rec)


def plot_gpi(row, use_log=False, ax=None):
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    gpi_values = gpi.genesis_potential_index(ds)

    print(row['Latitude'], row['Longitude'])
    other_tc_locations = literal_eval(row['Other TC Locations'])

    # Plot GPI values.
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    fig = ax.get_figure()
    yy, xx = np.meshgrid(ds.lon, ds.lat)
    if use_log:
        cs = ax.contourf(yy, xx, np.log(gpi_values + 1e-6))
    else:
        cs = ax.contourf(yy, xx, gpi_values)
    fig.colorbar(cs, ax=ax)

    # Not good, but usable.
    ax.set_ylim(5, 45)
    ax.set_xlim(100, 260)
    ax.set_title(f'GPI for {row["Date"]}')

    # Plot rectangles indicating location of tropical cyclogenesis
    # as well as other tropical cyclones.
    plot_rectangle((row['Longitude'], row['Latitude']), ax=ax)
    for loc in other_tc_locations:
        plot_rectangle(loc[::-1], ax=ax, color='orange')

    fig.tight_layout()
    return fig


# + tags=[]
for i in range(10):
    fig = plot_gpi(label_with_tc_df.iloc[i], use_log=False)
    display(fig)
    plt.close(fig)

# + tags=[]
for _, row in label_without_tc_df.sample(10).iterrows():
    fig = plot_gpi(row, use_log=False)
    display(fig)
    plt.close(fig)
# -


# ### Multiple Observations

# +
import os # noqa
from datetime import datetime, timedelta # noqa
import tc_formation.genesis_potential.genesis_potential_index as gpi # noqa


NC_FILENAME_FORMAT = 'fnl_%Y%m%d_%H_%M.nc'


def plot_something(row, previous_hours=[], use_log=False):
    path = row['Path']
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    date = datetime.strptime(basename, NC_FILENAME_FORMAT)
    print(date)

    print(row['Latitude'], row['Longitude'])
    other_tc_locations = literal_eval(row['Other TC Locations'])

    # nb_plots = len(previous_hours) + 2
    accumulated_gpi = None

    # Calculate and display the GPI of past observations.
    for prev_hour in previous_hours:
        prev_date = date - timedelta(hours=prev_hour)
        prev_obs_path = os.path.join(dirname, prev_date.strftime(NC_FILENAME_FORMAT))
        if not os.path.isfile(prev_obs_path):
            print(f'WARN: file {prev_obs_path} not found! SKIPPP!')

        ds = xr.load_dataset(prev_obs_path, engine='netcdf4')
        gpi_values = gpi.genesis_potential_index(ds)

        if accumulated_gpi is None:
            accumulated_gpi = np.ones_like(gpi_values, dtype=np.float64)

        accumulated_gpi *= gpi_values

        # Plot the GPI values.
        fig, ax = plt.subplots(figsize=(10, 6))
        yy, xx = np.meshgrid(ds.lon, ds.lat)
        # cs = ax.contourf(yy, xx, gpi_values)
        if use_log:
            cs = ax.contourf(yy, xx, np.log(gpi_values + 1e-6))
        else:
            cs = ax.contourf(yy, xx, gpi_values)
        ax.set_title(f'GPI values on {prev_date}')
        fig.colorbar(cs, ax=ax)

        plot_rectangle((row['Longitude'], row['Latitude']), ax=ax)
        for loc in other_tc_locations:
            plot_rectangle(loc[::-1], ax=ax, color='orange')

        display(fig)
        plt.close(fig)

    # Display the GPI of the current observation.
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    gpi_values = gpi.genesis_potential_index(ds)
    accumulated_gpi *= gpi_values

    # Plot the GPI values.
    fig, ax = plt.subplots(figsize=(10, 6))
    yy, xx = np.meshgrid(ds.lon, ds.lat)
    # cs = ax.contourf(yy, xx, gpi_values)
    if use_log:
        cs = ax.contourf(yy, xx, np.log(gpi_values + 1e-6))
    else:
        cs = ax.contourf(yy, xx, gpi_values)
    ax.set_title(f'GPI values on {row["Date"]}')
    fig.colorbar(cs, ax=ax)

    plot_rectangle((row['Longitude'], row['Latitude']), ax=ax)
    for loc in other_tc_locations:
        plot_rectangle(loc[::-1], ax=ax, color='orange')

    # Display the accumulated gpi.
    fig, ax = plt.subplots(figsize=(10, 6))
    yy, xx = np.meshgrid(ds.lon, ds.lat)
    # cs = ax.contourf(yy, xx, accumulated_gpi)
    if use_log:
        cs = ax.contourf(yy, xx, np.log(accumulated_gpi + 1e-6))
    else:
        cs = ax.contourf(yy, xx, accumulated_gpi)
    ax.set_title(f'Accumulated GPI values on {row["Date"]}')
    fig.colorbar(cs, ax=ax)

    plot_rectangle((row['Longitude'], row['Latitude']), ax=ax)
    for loc in other_tc_locations:
        plot_rectangle(loc[::-1], ax=ax, color='orange')


# + tags=[]
plot_something(label_with_tc_df.iloc[15], range(48, 0, -6), use_log=True)
# -

plot_something(label_without_tc_df.iloc[50], range(48, 0, -6), use_log=True)
