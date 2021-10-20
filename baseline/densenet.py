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
import sys # noqa
sys.path.append('..') # noqa

import tensorflow as tf
import tensorflow.keras as keras
import tf_metrics as tfm
import data
# -

# Use Densenet

model = keras.applications.DenseNet121(
    input_shape=(41, 181, 5),
    weights=None,
    include_top=True,
    classes=1,
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
model.summary()

# Load our training and validation data.

downsampled_training = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_train',
    batch_size=64,
    negative_samples_ratio=3)
validation = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_val')

# # First stage
#
# Train the model on the down-sampled data.

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

# Then, we will test on the test dataset to see the baseline results.

testing = data.load_data(
    '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb_test')
model.evaluate(testing)

# # Second stage
#
# Train the model on full dataset.

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

predictions = model.predict(testing)
print(predictions)
model.evaluate(testing)
