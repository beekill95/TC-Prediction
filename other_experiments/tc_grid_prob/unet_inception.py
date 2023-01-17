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

# %cd ../..

from datetime import datetime
from tc_formation.models import unet_inception as unet_i
from tc_formation import tf_metrics as tfm
import tc_formation.data.time_series as ts_data
from tc_formation.losses.hard_negative_mining import hard_negative_mining
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import xarray as xr

# # Predict TC Formation using Grid Probability

# Configurations to run for this experiment.

# +
exp_name = 'tc_grid_prob_unet_inception'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
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
# subset = dict(
#     absvprs=[900, 800, 750, 500, 200],
#     rhprs=[900, 800, 750, 500, 200],
#     tmpprs=[900, 800, 750, 500, 200],
#     hgtprs=[900, 800, 750, 500, 200],
#     vvelprs=[900, 800, 750, 500, 200],
#     ugrdprs=[900, 800, 750, 500, 200],
#     vgrdprs=[900, 800, 750, 500, 200],
# )
# data_shape = (41, 161, 37)
# subset = None
# data_shape = (41, 161, 135)

use_softmax = True
# -

# Create U-Net model with normalization layer.

# Then, we load the training and validation dataset.

tc_avg_radius_lat_deg = 3
data_loader = ts_data.TropicalCycloneWithGridProbabilityDataLoader(
    data_shape=data_shape,
    tc_avg_radius_lat_deg=tc_avg_radius_lat_deg,
    subset=subset,
    softmax_output=use_softmax,
    smooth_gt=True,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=128,
    shuffle=True)
validation = data_loader.load_dataset(val_path, batch_size=128)


# After that, we will initialize the normalization layer,
# and compile the model.

# +
@hard_negative_mining
def hard_negative_mined_sigmoid_focal_loss(y_true, y_pred):
    fl = tfa.losses.SigmoidFocalCrossEntropy()
    return fl(y_true, y_pred)

@hard_negative_mining
def hard_negative_mined_binary_crossentropy_loss(y_true, y_pred):
    l = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return l(y_true, y_pred)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


# +
# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
model = unet_i.UnetInception(
    input_shape=data_shape,
    model_name='unet_inception',
    classifier_activation='sigmoid' if not use_softmax else 'softmax',
    output_classes=1 if not use_softmax else 2,
    decoder_shortcut_mode='concat')
model.summary()

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=hard_negative_mined_sigmoid_focal_loss,
    metrics=[
        'binary_accuracy',
        keras.metrics.Recall(class_id=1),
        keras.metrics.Precision(class_id=1),
        tfm.CustomF1Score(class_id=1),
        #tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        #tfm.PrecisionScore(from_logits=True),
        #tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ])
# -

# Finally, we can train the model!

epochs = 150
model.fit(
    training,
    epochs=epochs,
    validation_data=validation,
    shuffle=True,
    callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=50,
            restore_best_weights=True
        ),
    ]
)

for leadtime in [6, 12, 18, 24, 30, 36, 42, 48]:
    testing = data_loader.load_dataset(
        test_path,
        leadtimes=leadtime,
        batch_size=128
    )
    print(f'\n**** LEAD TIME: {leadtime}')
    model.evaluate(testing)

# # Some Predictions

# +
import matplotlib.pyplot as plt # noqa
from mpl_toolkits.basemap import Basemap # noqa
import numpy as np # noqa
import pandas as pd # noqa
from tc_formation.data.data import load_observation_data_with_tc_probability # noqa
from tc_formation.plots import decorators, observations as plt_obs # noqa

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

def plot_groundtruth_and_prediction(tc_df):
    for _, row in tc_df.iterrows():
        dataset = xr.open_dataset(row['Path'])
        inp = {
            'Path': tf.constant(row['Path']),
            'TC': tf.constant(1 if row['TC'] else 0),
            'Latitude': tf.constant(row['Latitude'], dtype=float),
            'Longitude': tf.constant(row['Longitude'], dtype=float)
        }
        data, groundtruth = load_observation_data_with_tc_probability(
            inp,
            tc_avg_radius_lat_deg=tc_avg_radius_lat_deg,
            subset=subset)

        prediction = model.predict(np.asarray([data]))[0]
        
        fig, ax = plt.subplots(nrows=2, figsize=(30, 18))
        plot_tc_occurence_prob(dataset=dataset, prob=np.squeeze(groundtruth), ax=ax[0])
        plt_obs.plot_wind(dataset=dataset, pressure_level=800, skip=4, ax=ax[0])
        ax[0].set_title('Groundtruth')

        plot_tc_occurence_prob(dataset=dataset, prob=np.squeeze(prediction), ax=ax[1])
        plt_obs.plot_wind(dataset=dataset, pressure_level=800, skip=4, ax=ax[1])
        ax[1].set_title('Prediction')
        
        if row['TC']:
            title = f"""Prediction on date {row['Date']}
                        for tropical cyclone appearing on {row['First Observed']}"""
        else:
            title = f"Prediction on date {row['Date']}"
        fig.suptitle(title)
        fig.tight_layout()
        display(fig)
        plt.close(fig)
        
        print("=====\n=====\n====\n")
# -

# ## With TC

test_df = pd.read_csv(test_path)
test_with_tc_df = test_df[test_df['TC']].sample(5)
plot_groundtruth_and_prediction(test_with_tc_df)

# ## Without TC

test_without_tc_df = test_df[~test_df['TC']].sample(5)
plot_groundtruth_and_prediction(test_without_tc_df)
