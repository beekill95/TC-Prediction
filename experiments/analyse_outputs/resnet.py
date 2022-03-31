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

from tc_formation import plot
from tc_formation.data import data
import tc_formation.models.layers
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
from tc_formation.plots import observations as plt_obs
from tc_formation.plots import decorators as _d
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np
import sklearn.metrics as skmetrics
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import maskoceans

# # Plotting Results of ResNet 18 on Multiple Leadtime and Large Domain


model_path = 'outputs/baseline_resnet_multileadtime_2022_Jan_16_10_26_1st_ckp'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
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

full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
testing = data.load_data_v1(test_path, data_shape=data_shape, subset=subset)

# +
normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)


def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
testing = testing.map(normalize_data)
# -

# Using the model to predict the testing dataset.
pred = model.predict(testing)
pred = 1 / (1 + np.exp(-pred))

# +
y_true = np.concatenate([y for _, y in iter(testing)])

for threshold in np.arange(0.0, 1.0, 0.1):
    y_pred = np.where(pred >= threshold, 1, 0)

    print('\nAt threshold = ', threshold)
    print('\t F1 score: ', skmetrics.f1_score(y_true, y_pred))
    print('\t Precision: ', skmetrics.precision_score(y_true, y_pred))
    print('\t Recall: ', skmetrics.recall_score(y_true, y_pred))

# +
test_tc = pd.read_csv(test_path)
test_tc['Predicted'] = np.where(pred >= threshold, 1, 0).flatten()

matched = test_tc[test_tc['TC'] == test_tc['Predicted']]
difference = test_tc[test_tc['TC'] != test_tc['Predicted']]


# -

# # Plot Stuffs

# +
@_d._with_axes
@_d._with_basemap
def plot_SST(dataset, basemap=None, ax=None, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contour(lats, longs, dataset['tmpsfc'], levels=np.arange(270, 310, 2), cmap='Reds')
    ax.clabel(cs, inline=True, fontsize=20)

def plot_stuff(ds, pressure_level, ax):
    # Plot Relative Humidity
    plt_obs.plot_variable(
        dataset=ds,
        variable='rhprs',
        pressure_level=pressure_level,
        cmap='Blues',
        ax=ax,
        contourf_kwargs=dict(levels=np.arange(0, 110, 5)))
    
    # Plot wind field.
    plt_obs.plot_wind(dataset=ds, pressure_level=pressure_level, ax=ax, skip=4)

    # Plot SST
    plot_SST(dataset=ds, ax=ax)


# -

# # False Positives

false_positives = difference[difference['TC'] == 0].sample(5)

# +
false_positive_ds = []

for _, row in false_positives.iterrows():
    ds = xr.open_dataset(row['Path'])
    false_positive_ds.append(data.extract_variables_from_dataset(ds, subset))

    fig, axs = plt.subplots(nrows=2, figsize=(30, 16))

    fig.suptitle(
        f'False Positive: Wind field observed on {row["Date"]}')

    axs[0].set_title('Wind Field, SST, and RH at 850mb')
    plot_stuff(ds, 850, ax=axs[0])
    
    axs[1].set_title('Wind Field, SST, and RH at 500mb')
    plot_stuff(ds, 500, ax=axs[1])
    
    fig.tight_layout()
    display(fig)
    plt.close(fig)
# -

# # False Negatives

false_negatives = difference[difference['TC'] == 1].sample(5)

# +
false_negative_ds = []

for _, row in false_negatives.iterrows():
    ds = xr.open_dataset(row['Path'])
    false_negative_ds.append(data.extract_variables_from_dataset(ds, subset))

    fig, axs = plt.subplots(nrows=2, figsize=(30, 16), sharex=True)

    fig.suptitle(
        f'''False Negative: Wind field observed on {row["Date"]},
            there will be TC at {row["Latitude"]} lat, {row["Longitude"]} lon  on {row["First Observed"]}''')
    
    axs[0].set_title('Wind Field, SST, and RH at 850mb')
    plot_stuff(ds, 850, ax=axs[0])
    
    axs[1].set_title('Wind Field, SST, and RH at 500mb')
    plot_stuff(ds, 500, ax=axs[1])
    
    fig.tight_layout()
    display(fig)
    plt.close(fig)
