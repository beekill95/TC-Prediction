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

from datetime import datetime
from tc_formation.models import resnet
from tc_formation import tf_metrics as tfm
from tc_formation.metrics.bb import BBoxesIoUMetric
import tc_formation.data.time_series as ts_data
from tc_formation.losses.hard_negative_mining import hard_negative_mining
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import xarray as xr

# # Predict TC Formation with Location

# Configurations to run for this experiment.

exp_name = 'tc_loc_resnet'
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

# Create ResNet18 model with normalization layer.

input_layer = keras.Input(data_shape)
normalization_layer = preprocessing.Normalization()
model = resnet.ResNet18(
    input_tensor=normalization_layer(input_layer),
    classifier_activation=None,
    classes=3)
model.summary()

# Then, we load the training and validation dataset.

data_loader = ts_data.TropicalCycloneWithLocationDataLoader(
    data_shape=data_shape,
    subset=subset,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=128,
    shuffle=True)
validation = data_loader.load_dataset(val_path, batch_size=128)

# After that, we will initialize the normalization layer,
# and compile the model.

features = training.map(lambda feature, _: feature)
normalization_layer.adapt(features)

# Then, we will create loss function,
# metrics and compile the model.

# +
def loss(y_true, y_pred):
    fl = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
    mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    return fl(y_true[:, 0], y_pred[:, 0]) + y_true[:, 0] * mse(y_true[:, 1:], y_pred[:, 1:])


def squared_distance(y_true, y_pred):
    mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    return y_true[:, 0] * mse(y_true[:, 1:], y_pred[:, 1:])

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=[
        tfm.FromLogitsDecorator(keras.metrics.Recall(class_id=0)),
        tfm.FromLogitsDecorator(keras.metrics.Precision(class_id=0)),
        tfm.FromLogitsDecorator(tfm.CustomF1Score(class_id=0, name='f1')),
        squared_distance,
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
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=20,
            restore_best_weights=True
        ),
    ]
)

