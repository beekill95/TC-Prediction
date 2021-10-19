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
import tf_metrics as tfm
import data

# Build simple model as our baseline.

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(41, 181, 5)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
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
model.fit(training, epochs=epochs, validation_data=validation,
          class_weight={1: 3., 0: 1.})

# After the model is trained, we will test it on test data.

testing = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_test')
predictions = model.predict(testing)
print(predictions)
model.evaluate(testing)
