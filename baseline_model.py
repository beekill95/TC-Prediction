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

import tensorflow as tf
import tensorflow.keras.layers as layers
import data

# Build simple model as our baseline.

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(41, 181, 5)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dense(2048, activation='relu'),
    layers.Dense(1),
])
model.summary()

# Then, build the model with cross validation loss,
# and with adam optimizer.

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['binary_accuracy']
)

# Load our training data.

dataset = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb')

# Finally, train the model on the data.

# +
import numpy as np

epochs = 10
model.fit(dataset, epochs=epochs)
# -


