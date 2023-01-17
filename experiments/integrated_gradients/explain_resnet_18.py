# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
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

model_path = 'outputs/baseline_resnet_single_leadtime_2022_Jan_24_10_42_1st_ckp/'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

# ## Data Loading

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h.csv'
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
import matplotlib.pyplot as plt # noqa
from matplotlib.patches import Rectangle # noqa
from tc_formation.plots import observations as plt_obs # noqa
from tc_formation.plots import decorators as _d # noqa
from tc_formation.model_explanation import integrated_gradient as IG # noqa
from tc_formation.plots.integrated_gradient_visualizer import IntegratedGradientVisualizer # noqa

size = '30'
params = {'legend.fontsize': size,
         'axes.labelsize': size,
         'axes.titlesize': size,
         'xtick.labelsize': size,
         'ytick.labelsize': size}
plt.rcParams.update(params)

@_d._with_axes
@_d._with_basemap
def plot_SST(dataset, basemap=None, ax=None, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contour(lats, longs, dataset['tmpsfc'], levels=np.arange(270, 310, 2), cmap='Reds')
    ax.clabel(cs, inline=True, fontsize=20)
    
def plot_rectangle(center, ax, color='blue', size=5):
    half_size = size / 2.0
    center = np.asarray(center)
    rec = Rectangle(center - half_size, size, size, color=color, fill=False, lw=8.)
    ax.add_patch(rec)

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
import xarray as xr # noqa

def plot_samples(df):
    visualizer = IntegratedGradientVisualizer()

    for _, row in df.iterrows():
        ds = xr.open_dataset(row['Path'])
        print(row['Path'], row['Pred Prob'], row['Predicted'])
        print('First Observed: ', row['First Observed'])
        print(f'Location: {row["Latitude"]} lat - {row["Longitude"]} lon')

        # Load data, predict and calculate integrated gradient.
        X = tcdata.extract_variables_from_dataset(ds, subset)
        igrads = IG.integrated_gradient(aug_model, X, baseline_input, preprocessor=normalizer)
        X = normalizer(np.asarray([X]))
        a = model.predict(X)
        print(1/(1 + np.exp(-a)))
        print('Model Prediction: ', aug_model.predict(X))

        # Plot stuffs.
        fig, ax = plt.subplots(figsize=(30, 18))
        ax.set_title(f'SST at 850mb on date {row["Date"]}\nand Model Spatial Attribution for prediction on date {row["First Observed"]}')

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
        plot_rectangle((row["Longitude"], row["Latitude"]), size=10., ax=ax, color='darkorange')

        # Display the resulting plot.
        fig.tight_layout()
        display(fig)
        plt.close(fig)


# -

plot_samples(true_postivies.head(10))

plot_samples(true_postivies.tail(10))
