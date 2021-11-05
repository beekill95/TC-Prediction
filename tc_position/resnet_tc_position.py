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
sys.path.append('..')  # noqa

import data
import models.layers
import models.resnet
import tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import plot
import losses.mse_binary_loss
# -

# # ResNet with TC position
#
# The idea of this notebook is to experiment whether if we tell the DL model to learn
# additional information about the position of the TCs when they appear,
# it will perform better?
#
# In order to realize this idea, I will tell the ResNet to output 3 terms,
# The first term is whether it think there is a TC in the given day,
# the second term and third term are the latitude and longitude of the TC.

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
train_path = f'{data_path}_train'
val_path = f'{data_path}_val'
test_path = f'{data_path}_test'
data_shape = (41, 181, 13)

model = models.resnet.ResNet18(
    input_shape=data_shape,
    include_top=True,
    classes=3,
    # Turning of classifier activation allow us to treat the output
    # as whatever we want.
    # Here, we'll treat the first to be the classification,
    # the second and the third to be latitude and longitude.
    classifier_activation=None,)
model.summary()

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    # TODO: The loss will be custom!
    # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    loss=lambda y_true, y_pred: losses.mse_binary_loss.mse_focal_loss(
        y_true, y_pred, 5),
    # loss=keras.losses.MeanSquaredError(),
    metrics=[
        # 'mean_squared_error'
        tfm.NthBinaryAccuracy(name='binary_accuracy'),
        tfm.NthRecallScore(from_logits=True, name='recall_score'),
        tfm.NthPrecisionScore(from_logits=True, name='precision_score'),
        tfm.NthF1Score(num_classes=1, from_logits=True,
                       threshold=0.5, name='f1_score'),
    ]
)

# Load our training and validation data.

full_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    include_tc_position=True
)
downsampled_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    negative_samples_ratio=1,
    include_tc_position=True
)
validation = data.load_data(
    val_path, data_shape=data_shape, include_tc_position=True)

normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)
normalizer


# +
def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(normalize_data)
# -

# # First stage
#
# train the model on the down-sampled data.

# +
epochs = 75
first_stage_history = model.fit(
    downsampled_training,
    epochs=epochs,
    validation_data=validation,
    #class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_f1_score',
                    mode='max',
                    verbose=1,
                    patience=20,
                    restore_best_weights=True),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data(test_path,
                         data_shape=data_shape,
                         include_tc_position=True)
testing = testing.map(normalize_data)
model.evaluate(testing)

# # Second stage
#
# train the model on full dataset.

# +
second_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    #class_weight={1: 10., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ])


plot.plot_training_history(second_stage_history, "")
# -

# After the model is trained, we will test it on test data.

model.evaluate(testing)
