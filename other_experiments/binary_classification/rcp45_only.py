# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %cd ../..
# %load_ext autoreload
# %autoreload 2

from collections import OrderedDict
from tc_formation.binary_classifications.data.binary_classification_data_loader import BinaryClassificationDataLoader
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# -

# This experiment will be run on NCEP/FNL dataset
# and run prediction on future data.

# +
subset = OrderedDict(
    absvprs=(900, 750),
    rhprs=(750,),
    tmpprs=(900, 500),
    hgtprs=(500,),
    vvelprs=(500,),
    ugrdprs=(800, 200),
    vgrdprs=(800, 200),
    # tmpsfc=True,
    slp=True,
)
input_shape = (30, 30, 12)
dataloader = BinaryClassificationDataLoader((30, 30), subset)
future_rcp45_path = 'data/theanh_WPAC_RCP45_binary'

rcp45_train_ds, rcp45_val_ds, rcp45_test_ds = dataloader.load_dataset(
    future_rcp45_path, val_split=0.1, test_split=0.8, shuffle=False)
# for X, y in iter(ncep_ds):
#     print(X)
#     break

# +
def set_shape(shape):
    def _set_shape(X, y):
        X.set_shape(shape)
        return X, y

    return _set_shape

rcp45_train_ds = rcp45_train_ds.map(set_shape((None,) + input_shape))
rcp45_val_ds = rcp45_val_ds.map(set_shape((None,) + input_shape))
rcp45_test_ds = rcp45_test_ds.map(set_shape((None,) + input_shape))

# +
# def replace_nan_with_mean(X, y):
#     nonnan_mask = tf.math.is_finite(X)
#     layer_means = tf.reduce_mean(tf.ragged.boolean_mask(X, nonnan_mask), axis=(1, 2), keepdims=True)
#     print(nonnan_mask.shape, X.shape, (tf.ones_like(X) * layer_means).shape)
#     tf.print(nonnan_mask.shape, X.shape, (tf.ones_like(X) * layer_means).shape)
#     X = tf.where(nonnan_mask, X, tf.ones_like(X) * layer_means)
#     return X.to_tensor(), y
    

# rcp45_ds = rcp45_ds.map(replace_nan_with_mean)

# +
# for X, y in iter(rcp45_ds):
#     print(X.shape)
# -

# ## Model

# +
preprocessing = keras.Sequential([
    layers.Normalization(axis=-1),
])

preprocessing.layers[0].adapt(rcp45_train_ds.map(lambda X, _: X))
# -

model = keras.Sequential([
    layers.Input(input_shape),
    preprocessing,
    layers.Conv2D(
        64, 3,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(1e-4)),
    layers.MaxPool2D(2, 2),
    # layers.Conv2D(512, 3, activation='relu'),
    # layers.MaxPool2D(2, 2),
    # layers.Conv2D(1024, 3, activation='relu'),
    # layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        'binary_accuracy',
    ]
)
model.summary()

model.fit(
    rcp45_train_ds,
    epochs=100,
    validation_data=rcp45_val_ds,
    # class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
        # keras.callbacks.EarlyStopping(
        #     monitor='val_f1_score',
        #     mode='max',
        #     verbose=1,
        #     patience=10,
        #     restore_best_weights=True),
    ]
)

model.evaluate(rcp45_test_ds)
