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
from tc_formation.binary_classifications.data.patches_tfrecords_data_loader import PatchesTFRecordDataLoader
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tqdm.auto import tqdm
# -

# This experiment will be run on NCEP/FNL dataset
# and run prediction on the test NCEP/FNL data.
#
# The purpose of this experiment is just to see
# when the model is trained with all NCEP data,
# can it performs well on the same test data?

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
input_shape = (30, 30, 13)
dataloader = BinaryClassificationDataLoader((30, 30), subset)
ncep_path = 'data/binary_datasets/ncep_WP_binary_6h'

ncep_train_ds, ncep_val_ds, _ = dataloader.load_dataset(ncep_path, val_split=0.1)

# +
def set_shape(shape):
    def _set_shape(X, y):
        X.set_shape(shape)
        return X, y

    return _set_shape

ncep_train_ds = ncep_train_ds.map(set_shape((None,) + input_shape))
ncep_val_ds = ncep_val_ds.map(set_shape((None,) + input_shape))
# -

train_ds = ncep_train_ds
val_ds = ncep_val_ds

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
    validation_data=val_ds,
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

# ## Perform Prediction on Patches Dataset 

# + tags=[]
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


# + tags=[]
from tc_formation.binary_classifications.data.patches_tfrecords_data_loader import PatchesTFRecordDataLoader # noqa
import numpy as np # noqa


dataloader = PatchesTFRecordDataLoader()
test_path = 'data/patches_ncep_WP_EP_new_2_test_only/data_30_30.tfrecords'
test_patches_ds = dataloader.load_dataset(test_path, batch_size=128)
results_df = perform_prediction_on_patches(model, test_patches_ds)
results_df.to_csv('ncep_6h_test_only.csv', index=False)
