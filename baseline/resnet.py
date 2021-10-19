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
import sys
sys.path.append('..')

import data
import tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
# -

# Use ResNet

model = keras.applications.ResNet50(
    input_shape=(41, 181, 5),
    weights=None,
    include_top=True,
    classes=1,
    classifier_activation=None,
)

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ]
)

# Load our training and validation data.

downsampled_training = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_train',
    batch_size=64,
    negative_samples_ratio=3)
validation = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_val')

# First stage: train the model on the down-sampled data.

epochs = 50
model.fit(
    downsampled_training.shuffle(128),
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 3., 0: 1.},
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

# Second stage: train the model on full dataset.

full_training = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_train',
    batch_size=64,
)
model.fit(
    full_training.shuffle(128),
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 3., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ])

# After the model is trained, we will test it on test data.

testing = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_test')
predictions = model.predict(testing)
print(predictions)
model.evaluate(testing)
