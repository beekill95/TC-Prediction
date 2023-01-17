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

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tc_formation.data.data as tcdata

# # Explain ResNet model using Integrated Gradients

# ## Model Loading

model_path = 'outputs/baseline_resnet_vortex_removed_2022_Mar_31_10_26_1st_ckp/'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

# ## Data Loading

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/tc_ibtracs_12h_WP_EP_v4.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')
subset = dict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
)
data_shape = (41, 161, 13)

full_training = tcdata.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=False,
    subset=subset,
    group_same_observations=False,
)
testing = tcdata.load_data_v1(test_path, data_shape=data_shape, subset=subset, shuffle=False)

normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)


# ## Model Results

# +
def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
testing = testing.map(normalize_data)
# -

# Using the model to predict the testing dataset.
pred = model.predict(testing)
pred = 1 / (1 + np.exp(-pred))

pred[:5]

# +
test_tc = pd.read_csv(test_path)
test_tc['Pred Prob'] = pred
test_tc['Predicted'] = test_tc['Pred Prob'] >= 0.5

matched = test_tc[test_tc['TC'] == test_tc['Predicted']]
difference = test_tc[test_tc['TC'] != test_tc['Predicted']]

true_postivies = matched[matched['TC'] == 1]
true_postivies.head()
# -

# ## Baseline Input
# In order for Integrated Gradient to work,
# we have to find a baseline input with property that $F(x_0) \approx 0$
#
# In order to do that, we will try to follow the work of [MetNet-2](https://arxiv.org/abs/2111.07470),
# which is generate a baseline input where each feature values are the smallest one.
# After that, we'll have to verify indeed that this baseline input return a prediction near or approximately 0.

min_elem_iter = (full_training
    .map(lambda X, _: X)
    .map(lambda X: tf.reduce_min(X, axis=[0, 1, 2], keepdims=True))
    .map(lambda X: tf.squeeze(X, axis=0))
    .batch(2048) # Very large so we can take all the batch.
    .map(lambda X: tf.reduce_min(X, axis=0))
    .take(1)
    .as_numpy_iterator())
min_elem = next(min_elem_iter)
print(min_elem.shape)
print(min_elem)

# After we get the minimum value along each channel (environmental variables at different pressure level),
# we then  create a baseline input with the same dimension with the input to the model.
# Basically, we just want to repeat the those values along the width and height dimension to obtain an input with shape (41, 181, 13).

baseline_input = np.ones(data_shape) * min_elem
baseline_input[:5, :5]

# Finally, we just have to verify that this input produces prediction that is close to zero.

pred = model.predict(normalizer(np.asarray([baseline_input])))
1 / (1 + np.exp(-pred))

# This baseline doesn't work as it returns 1.
# We'll fall back to get zeros as input.

baseline_input = np.zeros(data_shape)
pred = model.predict(normalizer(np.asarray([baseline_input])))
1 / (1 + np.exp(-pred))

# ## Integrated Gradient
# ### Augmented Model to Return Gradient

sigmoid = 1 / (1 + tf.exp(-model.outputs[0]))
aug_model = keras.Model(model.inputs, sigmoid)

# +
import matplotlib.pyplot as plt
from tc_formation.plots import observations as plt_obs # noqa
from tc_formation.plots import decorators as _d # noqa
from tc_formation.model_explanation import integrated_gradient as IG # noqa
from tc_formation.plots.integrated_gradient_visualizer import IntegratedGradientVisualizer # noqa

@_d._with_axes
@_d._with_basemap
def plot_SST(dataset, basemap=None, ax=None, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contour(lats, longs, dataset['tmpsfc'], levels=np.arange(270, 310, 2), cmap='Reds')
    ax.clabel(cs, inline=True, fontsize=20)

def plot_stuff(ds, pressure_level, ax):
    # Plot Relative Humidity
#     plt_obs.plot_variable(
#         dataset=ds,
#         variable='rhprs',
#         pressure_level=pressure_level,
#         cmap='Blues',
#         ax=ax,
#         contourf_kwargs=dict(levels=np.arange(0, 110, 5)))
    
    # Plot wind field.
    plt_obs.plot_wind(dataset=ds, pressure_level=pressure_level, ax=ax, skip=4)

    # Plot SST
    plot_SST(dataset=ds, ax=ax)


# +
from global_land_mask import globe
import xarray as xr # noqa


fig, ax = plt.subplots(figsize=(15, 10))
ds = xr.load_dataset('data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/fnl_20180713_00_00.nc', engine='netcdf4')
lon = np.where(ds.lon < 180.0, ds.lon, 360.0 - ds.lon)
yy, xx = np.meshgrid(lon, ds.lat)
ocean_mask = globe.is_ocean(xx, yy)
cs = ax.imshow(ocean_mask)
fig.colorbar(cs, ax=ax)
ax.invert_yaxis()
print(lon)

# +
import xarray as xr # noqa
from ast import literal_eval # noqa
from matplotlib.patches import Rectangle


def plot_rectangle(center, ax, color, size=5):
    half_size = size / 2.0
    center = np.asarray(center)
    print(center)
    rec = Rectangle(center - half_size, size, size, color=color, fill=False, lw=2.)
    ax.add_patch(rec)


def plot_samples(true_positives_df):
    visualizer = IntegratedGradientVisualizer()

    for _, row in true_positives_df.iterrows():
        ds = xr.open_dataset(row['Path'])
        print(row['Path'], row['Pred Prob'], row['Predicted'])
        print('First Observed: ', row['First Observed'])
        print(f'Location: {row["Latitude"]} lat - {row["Longitude"]} lon')
        other_tc_locs = literal_eval(row["Other TC Locations"])
        print(f'Other TC Locations: {other_tc_locs}')

        # Load data, predict and calculate integrated gradient.
        X = tcdata.extract_variables_from_dataset(ds, subset)
        igrads = IG.integrated_gradient(aug_model, X, baseline_input, preprocessor=normalizer)
        X = normalizer(np.asarray([X]))
        a = model.predict(X)
        print(1/(1 + np.exp(-a)))
        print('Model Prediction: ', aug_model.predict(X))

        # Plot stuffs.
        fig, ax = plt.subplots(figsize=(30, 18))
        ax.set_title('SST and RH at 850mb, and Model Spatial Attribution')

        # Plot integrated result.
        a = visualizer.visualize(
            dataset=ds,
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=95,
            clip_below_percentile=28,
            morphological_cleanup=True,
            outlines=False,
            ax=ax,
        )

        plot_stuff(ds, 850, ax=ax)

        # Plot rectangles to highlight locations of cyclogenesis and other TCs.
        plot_rectangle((row["Longitude"], row["Latitude"]), ax=ax, color='blue')
        for other_loc in other_tc_locs:
            plot_rectangle(other_loc[::-1], ax=ax, color='orange')

        # Display the resulting plot.
        fig.tight_layout()
        display(fig)
        plt.close(fig)


# + [markdown] tags=[]
# ## True Positives
# -

plot_samples(true_postivies.head(10))

plot_samples(true_postivies.tail(10))

# ## False Positives

false_positives = difference[difference['TC'] == 0]
plot_samples(false_positives.head(10))

# ## False Negatives

false_negatives = difference[difference['TC']]
plot_samples(false_negatives.head(10))
