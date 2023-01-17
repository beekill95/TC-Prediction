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

# %cd ../..
# %load_ext autoreload
# %autoreload 2

from datetime import datetime
from tc_formation.models import unet
from tc_formation import tf_metrics as tfm
import tc_formation.metrics.bb as bb
import tc_formation.data.time_series as ts_data
import tc_formation.utils.unet_track as unet_track
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import xarray as xr


# # Unet Spatial Statistics

data_path = 'data/theanh_WPAC_baseline/tc_12h.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
subset = dict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
)
subset = dict(
    hgtprs=[700, 500, 250],
    ugrdprs=[700, 500, 250],
    vgrdprs=[700, 500, 250],
    capesfc=None,
    absvprs=None,
    rhprs=None,
    tmpprs=None,
    vvelprs=None,
    tmpsfc=None,
    slp=None,
)
data_shape = (218, 434, 9)
use_softmax = False

# Reload the model from checkpoint.

# + tags=[]
model_path = 'outputs/tc_grid_prob_unet_baseline_12h_2022_Sep_13_10_34_ckp_best_val'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()


# -

# Then, our data.

# +
def remove_nans(x, y):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), y

def spatial_scale(x, y):
    return x[:, ::2, ::2], y[:, ::2, ::2]


tc_avg_radius_lat_deg = 3
data_loader = ts_data.TropicalCycloneWithGridProbabilityDataLoader(
    data_shape=data_shape,
    tc_avg_radius_lat_deg=tc_avg_radius_lat_deg,
    subset=subset,
    softmax_output=use_softmax,
    smooth_gt=True,
)
# -

# Don't mind the naming,
# this particular model is trained on train dataset,
# and we want to see what kind of performance on training data.
testing = data_loader.load_dataset(
    train_path,
    batch_size=64,
).map(spatial_scale).map(remove_nans)
predictions = model.predict(testing)

# +
import matplotlib.pyplot as plt # noqa
from mpl_toolkits.basemap import Basemap # noqa
import matplotlib.patches as patches # noqa
import numpy as np # noqa
import pandas as pd # noqa
import tc_formation.data.label as label # noqa
from tc_formation.data.data import load_observation_data_with_tc_probability # noqa
from tc_formation.plots import decorators, observations as plt_obs # noqa
import xarray as xr # noqa

size = '30'
params = {'legend.fontsize': size,
         'axes.labelsize': size,
         'axes.titlesize': size,
         'xtick.labelsize': size,
         'ytick.labelsize': size}
plt.rcParams.update(params)


@decorators._with_axes
@decorators._with_basemap
def plot_tc_occurence_prob(
        dataset: xr.Dataset,
        prob: np.ndarray,
        basemap: Basemap = None,
        *args, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contourf(lats, longs, prob, cmap='OrRd', levels=np.arange(0, 1.01, 0.05))
    basemap.colorbar(cs, "right", size="5%", pad="2%")

# Similar to the above.
test_df = pd.read_csv(train_path)
with_tc_mask = test_df['TC'] == 1
test_df_with_tc = test_df[with_tc_mask]
center_locator = unet_track.UnetPredictionCenter()
pred_range = slice(50, 65)
for path, pred, lat, lon in zip(test_df_with_tc['Path'][pred_range],
                                predictions[with_tc_mask][pred_range],
                                test_df_with_tc['Latitude'][pred_range],
                                test_df_with_tc['Longitude'][pred_range]):
    centers = center_locator.get_centers(pred)
    distribution = unet_track.tc_formation_spatial_distribution(
        domain_size=(114, 219),
        centers=centers,
    )
    print(path, centers)
    print('truth', lat, lon)

    datasets = xr.load_dataset(path)
    datasets = datasets.sel(dict(lat=datasets['lat'][::2], lon=datasets['lon'][::2]))
    plot_tc_occurence_prob(dataset=datasets, prob=pred.squeeze())
    plt.show()
    plt.close()
# -

# Now, I will calculate yearly spatial distribution of TC formation.
# First, we must extract the year from test dataframe.

# +
from datetime import datetime # noqa

test_df['Year'] = test_df['Date'].apply(lambda d: datetime.strptime(d, '%Y-%m-%d %H:%M:%S').year)
# -

# Next, we will show the spatial distribution of these predictions.

# +
@decorators._with_axes
@decorators._with_basemap
def plot_tc_formation_predicted_spatial_distribution(
        dataset: xr.Dataset,
        distribution: np.ndarray,
        basemap: Basemap = None,
        *args, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contourf(lats, longs, distribution)
    basemap.colorbar(cs, "right", size="5%", pad="2%")


cur_year = test_df['Year'].iloc[0]
tc_dist = np.zeros((109, 217))
for path, pred, year in zip(test_df['Path'], predictions, test_df['Year']):
    if year != cur_year:
        # Load the dataset to get lat and lon.
        ds = xr.load_dataset(path)
        ds = ds.sel(dict(lat=ds['lat'][::2], lon=ds['lon'][::2]))

        # Plot the result
        plot_tc_formation_predicted_spatial_distribution(dataset=ds, distribution=tc_dist)
        plt.title(f'Spatial Distribution for year {year}.')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Reset the dist.
        tc_dist = np.zeros((109, 217))

        # Move to new year.
        cur_year = year

    # Obtain the center from the prediction,
    # then increase the count in `tc_dist`.
    centers = center_locator.get_centers(pred)
    distribution = unet_track.tc_formation_spatial_distribution(
        domain_size=(109, 217),
        centers=centers,
    )
    tc_dist += distribution
