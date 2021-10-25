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
import tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import plot
# -

# Use ResNet

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb'
data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels/6h_700mb'
train_path = f'{data_path}_train'
val_path = f'{data_path}_val'
test_path = f'{data_path}_test'
data_shape = (41, 181, 15)

model = keras.applications.ResNet50(
    input_shape=data_shape,
    weights=None,
    include_top=True,
    classes=1,
    classifier_activation=None,
)

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

full_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
)
downsampled_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    negative_samples_ratio=1)
validation = data.load_data(val_path, data_shape=data_shape)

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
epochs = 50
first_stage_history = model.fit(
    downsampled_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data(test_path, data_shape=data_shape)
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
    class_weight={1: 10., 0: 1.},
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
