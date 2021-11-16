# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.12
# ---

# %load_ext dotenv
# %dotenv

# +
import sys  # noqa
sys.path.append('../..')  # noqa

import tc_formation.plots.observations as obs_plt
import tc_formation.data as data
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
# -

# Load all the TC to see what we got.
data_home = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_700mb'
tc = pd.read_csv(os.path.join(data_home, 'tc.csv'), dtype={
    'Observation': 'str',
    'Genesis': 'str',
    'End': 'str'
})
tc[tc['TC'] == 1]

# +
# Of these storms, how many of them have positive latitude
# tc[(tc['TC'] == 1) & (tc['Latitude'] > 0)]

# +
# Of these storms, how many of them have latitude from 5 to 45
# tc[(tc['TC'] == 1) & (tc['Latitude'] <= -5) & (tc['Latitude'] >= -45)]

# +
# Of these storms, how many of them have latitude outside 5 to 45
#tc[(tc['TC'] == 1) & ~((tc['Latitude'] <= -5) & (tc['Latitude'] >= -45))]

# +
ds = xr.open_dataset(os.path.join(data_home, 'fnl_20210822_12_00.nc'))
lat = ds['lat']
lon = ds['lon']

ds

# +
levels = [800, 200]
nb_levels = len(levels)
fig, axs = plt.subplots(nrows=nb_levels, ncols=1, figsize=(30, 8 * nb_levels))

for lev, ax in zip(levels, axs):
    obs_plt.plot_variable(dataset=ds, variable='tmpsfc', ax=ax, contourf_kwargs=dict(levels=np.arange(270, 320, 2)))
    obs_plt.plot_wind(dataset=ds, pressure_level={'lev': lev}, ax=ax)
    ax.set_title(f'Wind field at {lev}mb & surface temperature')
# -

# We will randomly sample 5 date with tc, and 5 days without tc
# to see what are the different between them.

# +
tc_test = data.load_tc_with_observation_path(f'{data_home}_test')

has_tc = tc_test[tc_test['TC'] == 1].sample(5)
no_tc = tc_test[tc_test['TC'] == 0].sample(5)


def plot_observations_in_df(df, title):
    for _, df_row in df.iterrows():
        ds_ = xr.open_dataset(df_row['Path'])
        print(df_row['Path'], df_row['Genesis'], df_row['Latitude'], df_row['Longitude'])

        levels = [800, 200]
        nb_levels = len(levels)
        fig, axs = plt.subplots(nrows=nb_levels, ncols=1,
                                figsize=(30, 8 * nb_levels))

        fig.suptitle(title)
        for lev, ax in zip(levels, axs):
            obs_plt.plot_variable(dataset=ds_, variable='tmpsfc', ax=ax)
            obs_plt.plot_wind(dataset=ds_, pressure_level={'lev': lev}, ax=ax)
            ax.set_title(f'Wind field at {lev}mb & surface temperature')
            
        display(fig)
        plt.close(fig)


plot_observations_in_df(has_tc, 'Observations on date with TC')
# -

plot_observations_in_df(no_tc, 'Observations on date without TC')
