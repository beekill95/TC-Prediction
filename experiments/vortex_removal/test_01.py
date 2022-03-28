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
import numpy as np
import pandas as pd
from tc_formation.plots import observations as plt_obs
from tc_formation.plots import decorators as _d
from scipy import fft
import xarray as xr

# # Vortex Removal Test 01

# + [markdown] tags=[]
# ## Label Files
# -

data = pd.read_csv('data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv')
data.head()

# We will only care about observations with tropical cyclones.
obs_tc = data[data['Is Other TC Happening'] == True].reset_index()
obs_tc.sample(n=10)

# ## Observation with Tropical Cyclone

row_with_tc = obs_tc.iloc[2919]
row_with_tc.Path

obs = xr.load_dataset(row_with_tc['Path'])
obs

# +
fig, axes = plt.subplots(nrows=2, figsize=(30, 18))

ax = axes[0]
ax.set_title('U-wind at 925mb')
plt_obs.plot_variablef(dataset=obs, variable='ugrdprs', pressure_level=925, ax=ax)

ax = axes[1]
ax.set_title('V-wind at 925mb')
plt_obs.plot_variablef(dataset=obs, variable='vgrdprs', pressure_level=925, ax=ax)
# -

fig, ax = plt.subplots(figsize=(30, 9))
ax.set_title('Absolute vorticity at 500mb')
plt_obs.plot_variablef(dataset=obs, variable='absvprs', pressure_level=500, ax=ax)

# ## Tropical Cyclone removal

from ast import literal_eval # noqa
import tc_formation.vortex_removal.vortex_removal as vr # noqa

tc_loc = literal_eval(row_with_tc['Other TC Locations'])[0]
tc_loc

# Select region centered at TC center.
centered_tc = obs.loc[dict(lat=slice(round(tc_loc[0] - 5), round(tc_loc[0] + 5)),
                       lon=slice(round(tc_loc[1] - 5), round(tc_loc[1] + 5)))]
centered_tc

# +
fig, axes = plt.subplots(nrows=2, figsize=(6, 6))

ax = axes[0]
ax.set_title('U-wind at 925mb')
plt_obs.plot_variablef(dataset=centered_tc, variable='ugrdprs', pressure_level=925, ax=ax)

ax = axes[1]
ax.set_title('V-wind at 925mb')
plt_obs.plot_variablef(dataset=centered_tc, variable='vgrdprs', pressure_level=925, ax=ax)

# +
u_wind_925 = obs.vgrdprs.sel(lev=925).values
tc_loc_converted = [tc_loc[0] - np.min(obs.lat).values,
                    tc_loc[1] - np.min(obs.lon).values]
print(tc_loc_converted, tc_loc[0] - np.min(obs.lat).values)
field = vr.remove_vortex(u_wind_925, [tc_loc_converted], 10)

fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
ax = axes[0]
cs = ax.contourf(u_wind_925, cmap='rainbow')
ax.set_title('Original Field')
fig.colorbar(cs, ax=ax)

ax = axes[1]
cs = ax.contourf(field, cmap='rainbow')
ax.set_title('TC removed field')
fig.colorbar(cs, ax=ax)
# -

# Reassign values into original observation

# +
obs_tc_removed = obs.copy(deep=True)
tc_loc_converted = [tc_loc[0] - np.min(obs.lat).values,
                    tc_loc[1] - np.min(obs.lon).values]

u_wind_925 = obs.ugrdprs.sel(lev=925).values
field = vr.remove_vortex(u_wind_925, [tc_loc_converted], 10)
obs_tc_removed.ugrdprs.loc[dict(lev=925)] = field

v_wind_925 = obs.vgrdprs.sel(lev=925).values
field = vr.remove_vortex(v_wind_925, [tc_loc_converted], 10)
obs_tc_removed.vgrdprs.loc[dict(lev=925)] = field

fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
ax = axes[0]
plt_obs.plot_wind(dataset=obs, pressure_level=925, ax=ax)
ax.set_title('Original field')

ax = axes[1]
plt_obs.plot_wind(dataset=obs_tc_removed, pressure_level=925, ax=ax)
ax.set_title('Processed field')

# +
import tc_formation.vortex_removal.vortex_removal as vr # noqa

obs_tc_removed = vr.remove_vortex_ds(obs, [tc_loc], 10)
obs_tc_removed

fig, axes = plt.subplots(nrows=2, figsize=(15, 10))
ax = axes[0]
plt_obs.plot_wind(dataset=obs, pressure_level=925, ax=ax)
ax.set_title('Original field')

ax = axes[1]
plt_obs.plot_wind(dataset=obs_tc_removed, pressure_level=925, ax=ax)
ax.set_title('Processed field')
# -

obs_tc_removed.to_netcdf('after.nc')
