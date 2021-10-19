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

import sys
sys.path.append('..')
import tensorflow as tf
import tensorflow.keras as keras
import tf_metrics as tfm
import data

# Using VGG 16 model

model = keras.applications.VGG16(
    weights=None,
    input_shape=(41, 181, 5),
    include_top=True,
    classes=1,
    classifier_activation=None)
model.summary()

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=1, from_logits=True),
    ]
)

# Load our training and validation data.

training = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_train',
    batch_size=64,
    negative_samples_ratio=3)
validation = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_val')

# Train the model on the data.

epochs = 50
model.fit(training.shuffle(64), epochs=epochs, validation_data=validation,
          class_weight={1: 4., 0: 1.},
          shuffle=True
          )

# After the model is trained, we will test it on test data.

testing = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_test')
predictions = model.predict(testing)
print(predictions)
model.evaluate(testing)
