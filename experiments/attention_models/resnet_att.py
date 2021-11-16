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

from tc_formation import data, plot, tf_metrics as tfm
import tc_formation.models.resnet_att as resnet_att
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
# -

# Use ResNet

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

exp_name = 'attention_resnet'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb'
#data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/nolabels_wp_only_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h.csv'
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
data_shape = (41, 81, 13)

model = resnet_att.ResNet50Att(
    input_shape=data_shape,
    include_top=True,
    classes=1,
    classifier_activation=None,
    spatial_attention=True,
    channel_attention=False,
)
model.summary()

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ]
)

# Load our training and validation data.

full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
# downsampled_training = data.load_data_v1(
#     train_path,
#     data_shape=data_shape,
#     batch_size=64,
#     shuffle=True,
#     subset=subset,
#     negative_samples_ratio=1)
validation = data.load_data_v1(val_path, data_shape=data_shape, subset=subset)

normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)
normalizer


# +
def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
# downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(normalize_data)
# -

# # First stage
#
# train the model on the down-sampled data.

# +
epochs = 150
first_stage_history = model.fit(
    # downsampled_training,
    # testing with full training
    full_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 15., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=20,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp",
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
        ),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data_v1(test_path, data_shape=data_shape, subset=subset)
testing = testing.map(normalize_data)
model.evaluate(testing)

# # Second stage
#
# train the model on full dataset.

# +
# second_stage_history = model.fit(
#     full_training,
#     epochs=epochs,
#     validation_data=validation,
#     class_weight={1: 10., 0: 1.},
#     shuffle=True,
#     callbacks=[
#         keras.callbacks.EarlyStopping(
#             monitor='val_f1_score',
#             mode='max',
#             verbose=1,
#             patience=10,
#             restore_best_weights=True),
#     ])


# plot.plot_training_history(second_stage_history, "")
# -

# After the model is trained, we will test it on test data.

# model.evaluate(testing)
