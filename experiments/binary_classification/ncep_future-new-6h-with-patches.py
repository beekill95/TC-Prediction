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
import numpy as np
import pandas as pd
from tc_formation.binary_classifications.data.binary_classification_data_loader import BinaryClassificationDataLoader
from tc_formation.binary_classifications.data.patches_data_loader import PatchesDataLoader
from tc_formation.binary_classifications.data.patches_tfrecords_data_loader import PatchesTFRecordDataLoader
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tqdm.auto import tqdm
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
    tmpsfc=True,
    pressfc=True,
)
# corresponding_name = {
#     'pressfc': 'slp',
# }
# subset_theanh = OrderedDict(
#     (corresponding_name.get(key, key), val) for key, val in subset.items()
# )
# print(subset_theanh)
input_shape = (30, 30, 13)
dataloader = BinaryClassificationDataLoader((30, 30), subset)
# dataloader_theanh = BinaryClassificationDataLoader((30, 30), subset_theanh)
ncep_path = 'data/binary_datasets/ncep_WP_binary_6h'
future_rcp45_path = 'data/binary_datasets/WRF_RCP45_5_binary_6h'

ncep_train_ds, _, _ = dataloader.load_dataset(ncep_path,)
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

ncep_train_ds = ncep_train_ds.map(set_shape((None,) + input_shape))
# ncep_val_ds = ncep_val_ds.map(set_shape((None,) + input_shape))
rcp45_train_ds = rcp45_train_ds.map(set_shape((None,) + input_shape))
rcp45_test_ds = rcp45_test_ds.map(set_shape((None,) + input_shape))
rcp45_val_ds = rcp45_val_ds.map(set_shape((None,) + input_shape))
# -

train_ds = ncep_train_ds.concatenate(rcp45_train_ds)

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

preprocessing.layers[0].adapt(train_ds.map(lambda X, _: X))
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
    train_ds,
    epochs=100,
    validation_data=rcp45_val_ds,
    # class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ]
)

model.evaluate(rcp45_test_ds)

# ## Perform Prediction on Patches Dataset 

# + tags=[]
from tc_formation.binary_classifications.data.patches_data_loader import PatchesDataLoader
from scipy.special import expit


def perform_prediction_on_patches(model, patches_ds) -> pd.DataFrame:
    results = []
    # nb_batches = len(patches_ds)
    
    for X, coords, paths in tqdm(iter(patches_ds)):
        nb_patches = X.shape[0]
        # Perform prediction on these patches.
        # print('before')
        pred = expit(model.predict(X, verbose=False))
        # print('after')
        coords = coords.numpy()
        paths = paths.numpy()

        # Store the results.
        results.extend(
            (dict(path=paths[j].decode('utf-8'),
                  lat=coords[j, 0],
                  lon=coords[j, 1],
                  pred=pred[j, 0])
             for j in range(nb_patches)))

    # Sort the results.
    results = pd.DataFrame(results)
    results = results.sort_values(
        ['path', 'lat', 'lon'],
        axis=0,
        ascending=True,
        ignore_index=True)

    return results


# test_path = 'data/patches_RCP45_size_30_stride_5_0h'
# patches_dataloader = PatchesDataLoader(subset=subset, output_size=(30, 30))
# test_patches_ds = patches_dataloader.load_dataset(test_path, batch_size=128)
# results_df = perform_prediction_on_patches(model, test_patches_ds)
# results_df.to_csv('ncep_future_6h.csv')

# + tags=[]
from tc_formation.binary_classifications.data.patches_tfrecords_data_loader import PatchesTFRecordDataLoader # noqa
import numpy as np # noqa


dataloader = PatchesTFRecordDataLoader()
test_path = 'data/patches_RCP45_size_30_stride_5_0h/data_30_30.tfrecords'
test_patches_ds = dataloader.load_dataset(test_path, batch_size=128)
results_df = perform_prediction_on_patches(model, test_patches_ds)
results_df.to_csv('ncep_future_6h.csv', index=False)
