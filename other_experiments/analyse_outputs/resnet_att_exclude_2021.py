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

# +
import sys  # noqa
sys.path.append('../..')  # noqa

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as skmetrics
from tc_formation import data
from tc_formation.plots import observations as plt_obs
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import xarray as xr
# -

# # Analyzing ResNet Attention Output

# Path to the model's checkpoint that we want to analyze the output.

model_path = '../attention_models/outputs/attention_resnet_2021_Nov_06_16_35_1st_ckp/'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

# Path to the data that we want to inspect.

data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
train_path = f'{data_path}_train'
val_path = f'{data_path}_val'
test_path = f'{data_path}_test'
subset = dict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
)
data_shape = (41, 181, 13)

# Load the whole dataset to create a layer to normalize data.

full_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
testing = data.load_data(test_path, data_shape=data_shape, subset=subset)


# +
normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)
normalizer


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
test_tc = data.load_tc_with_observation_path(test_path)
test_tc['Predicted'] = np.where(pred >= threshold, 1, 0).flatten()

matched = test_tc[test_tc['TC'] == test_tc['Predicted']]
difference = test_tc[test_tc['TC'] != test_tc['Predicted']]
# -

print('True Positive')
with pd.option_context("display.min_rows", None, "display.max_rows", None,
                       "display.max_columns", None, 'display.max_colwidth', None):
    display(matched[matched['TC'] == 1])

# The model predicted yes, but actually no
print('False Positive')
with pd.option_context("display.min_rows", None, "display.max_rows", None,
                       "display.max_columns", None, 'display.max_colwidth', None):
    display(difference[difference['TC'] == 0])

# The model predicted no, but actually yes
print('False Negative')
with pd.option_context("display.min_rows", None, "display.max_rows", None,
                       "display.max_columns", None, 'display.max_colwidth', None):
    display(difference[difference['TC'] == 1])

# After different results given by the models,
# we will now look at how those models make prediction.
#
# ## Analyzing False Positive
#
# First, we'll choose randomly 5 samples of False Positive predictions,
# then, we'll see how the observations are.

false_positives = difference[difference['TC'] == 0].sample(5)

# +
false_positive_ds = []

for _, row in false_positives.iterrows():
    ds = xr.open_dataset(row['Path'])
    false_positive_ds.append(data.extract_variables_from_dataset(ds, subset))

    fig, axs = plt.subplots(nrows=2, figsize=(30, 16))

    fig.suptitle(
        f'False Positive: Wind field observed on {row["Observation"]}')

    axs[0].set_title('Wind field at 800mb and sea surface temperature')
    plt_obs.plot_variable(dataset=ds, variable='tmpsfc', ax=axs[0])
    plt_obs.plot_wind(dataset=ds, pressure_level=800, ax=axs[0])

    axs[1].set_title('Wind field at 200mb and RH at 750mb')
    plt_obs.plot_variable(dataset=ds, variable='rhprs',
                          pressure_level=750, ax=axs[1])
    plt_obs.plot_wind(dataset=ds, pressure_level=200, ax=axs[1])

    fig.tight_layout()
    display(fig)
    plt.close(fig)
# -

# ## Analyzing False Negative
#
# Similarly, we will randomly choose 5 samples to display

false_negatives = difference[difference['TC'] == 1].sample(5)

# +
false_negative_ds = []

for _, row in false_negatives.iterrows():
    ds = xr.open_dataset(row['Path'])
    false_negative_ds.append(data.extract_variables_from_dataset(ds, subset))

    fig, axs = plt.subplots(nrows=2, figsize=(30, 16), sharex=True)

    fig.suptitle(
        f'''False Negative: Wind field observed on {row["Observation"]},
            there will be TC at {row["Latitude"]} lat, {row["Longitude"]} lon''')

    axs[0].set_title('Wind field at 800mb and sea surface temperature')
    plt_obs.plot_variable(dataset=ds, variable='tmpsfc', ax=axs[0])
    plt_obs.plot_wind(dataset=ds, pressure_level=800, ax=axs[0])

    axs[1].set_title('Wind field at 200mb and RH at 750mb')
    plt_obs.plot_variable(dataset=ds, variable='rhprs',
                          pressure_level=750, ax=axs[1])
    plt_obs.plot_wind(dataset=ds, pressure_level=200, ax=axs[1])

    fig.tight_layout()
    display(fig)
    plt.close(fig)
