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
import sys  # noqa
sys.path.append('../..')  # noqa

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from tc_formation.data import load_observation_data_with_tc_probability, group_observations_by_date
from tc_formation.plots import decorators, observations
import xarray as xr
# -

# # Test Load data with Probability

data_path = '../../data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h.csv'
data: pd.DataFrame = pd.read_csv(data_path)
data = group_observations_by_date(data)

# Helper function to plot the given probability.

@decorators._with_axes
@decorators._with_basemap
def plot_tc_occurence_prob(
        dataset: xr.Dataset,
        prob: np.ndarray,
        basemap: Basemap = None,
        *args, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contourf(lats, longs, prob, cmap='OrRd')
    basemap.colorbar(cs, "right", size="5%", pad="2%")

# ## Observations with tropical cyclones

true_cases = data[data['TC']].sample(10)
for _, case in true_cases.iterrows():
    dataset = xr.load_dataset(case['Path'])
    _, prob = load_observation_data_with_tc_probability(case, tc_avg_radius_lat_deg=5, clip_threshold=0.05)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tc_occurence_prob(dataset=dataset, prob=prob, ax=ax)
    observations.plot_wind(dataset=dataset, pressure_level=800, ax=ax, skip=4)
    
    print(case['TC Id'], np.sum(prob > 0))

    ax.set_title(f'Tropical cyclone at {case["Latitude"]} degree latitude, {case["Longitude"]} degree longitude')
    display(fig)
    plt.close(fig)
