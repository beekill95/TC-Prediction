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
# %cd ../../..
# %load_ext autoreload
# %autoreload 2

import pandas as pd
from tc_formation.binary_classifications.data.patches_tfrecords_data_loader import PatchesTFRecordDataLoader
import tensorflow as tf
from tqdm.auto import tqdm


# -

# # Model
#
# Load pretrained model.

# +
class F1(tf.keras.metrics.Metric):
    def __init__(self, thresholds, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)

        self._precision = tf.keras.metrics.Precision(thresholds=thresholds)
        self._recall = tf.keras.metrics.Recall(thresholds=thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._precision.update_state(y_true, y_pred, sample_weight)
        self._recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        return 2 * p * r / (p + r + 1e-6)

    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()


path = 'saved_models/a_bit_future_RCP45_random_positive_no_pca_developed_storms_removed_v2_no_capesfc_leadtime_0h_f1_0.652'
model = tf.keras.models.load_model(path, compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-8),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        'binary_accuracy',
        tf.keras.metrics.Precision(thresholds=0.5),
        tf.keras.metrics.Recall(thresholds=0.5),
        F1(thresholds=0.5),
    ],
)
model.summary()
# -

# We probably want to train on some future data so that the model.

# +
from collections import OrderedDict # noqa
from tc_formation.binary_classifications.data.binary_classification_data_loader import BinaryClassificationDataLoader # noqa


def set_shape(shape, batch=True):
    def _set_shape(X, y):
        if batch:
            X.set_shape((None, ) + shape)
            # y.set_shape((None, 1))
        else:
            X.set_shape(shape)
            y.set_shape((1,))

        return X, y

    return _set_shape


pressure_levels = (
    1000, 975, 950, 925, 900, 850, 800, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
)
subset = OrderedDict(
    absvprs=pressure_levels,
    hgtprs=pressure_levels,
    pressfc=True,
    rhprs=pressure_levels,
    tmpprs=pressure_levels,
    tmpsfc=True,
    ugrdprs=pressure_levels,
    vgrdprs=pressure_levels,
    vvelprs=pressure_levels,
)
input_shape = (31, 31, 135)
dataloader_theanh = BinaryClassificationDataLoader((31, 31), subset)
future_rcp45_path = 'data/binary_datasets/WRF_RCP45_5_binary_0h'
rcp45_train_ds, rcp45_val_ds, _ = dataloader_theanh.load_dataset(
    future_rcp45_path,
    val_split=0.1,
    test_split=0.8,
    shuffle=False)

rcp45_train_ds = rcp45_train_ds.map(set_shape(input_shape))
rcp45_val_ds = rcp45_val_ds.map(set_shape(input_shape))

# + tags=[]
model.fit(
    rcp45_train_ds,
    epochs=1000,
    validation_data=rcp45_val_ds,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            verbose=1,
            patience=200,
            restore_best_weights=True),
    ]
)
# -

# Also, I want to know if it still works with the original data that it was trained on.

# +
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader # noqa


dataloader = PatchesWithGenesisTFRecordDataLoader()
test_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_varibles_no_capesfc_developed_storms_removed_v2_Test.tfrecords'
original_test_patches_ds = dataloader.load_dataset(test_path, batch_size=256)

input_shape = (31, 31, 135)
def set_shape(shape, batch=True):
    def _set_shape(X, y):
        if batch:
            X.set_shape((None, ) + shape)
            y.set_shape((None, 1))
        else:
            X.set_shape(shape)
            y.set_shape((1,))

        return X, y

    return _set_shape


original_test_patches_ds = original_test_patches_ds.map(set_shape(input_shape))

# +
from sklearn.metrics import precision_score, recall_score, f1_score # noqa
import numpy as np # noqa


yy_pred = []
yy_true = []
for X, y in tqdm(iter(original_test_patches_ds)):
    y_pred = model.predict(X, verbose=0)
    yy_pred.append(y_pred)
    yy_true.append(y)

yy_pred = np.concatenate(yy_pred, axis=0).flatten()
yy_true = np.concatenate(yy_true, axis=0).flatten()

# print(f'{precision_score(yy_true, yy_pred)=:.4f}, {recall_score(yy_true, yy_pred)=:.4f}, {f1_score(yy_true, yy_pred)=:.4f}')

# +
thresholds = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []
for threshold in thresholds:
    pred = np.where(yy_pred > threshold, 1, 0)
    results.append(dict(
        threshold=threshold,
        f1=f1_score(yy_true, pred),
        recall=recall_score(yy_true, pred),
        precision=precision_score(yy_true, pred),
    ))

results = pd.DataFrame(results)
results.plot.bar(x='threshold', figsize=(8, 6))
# -

results

# # Predictions on Future Projections
# ## RCP45

# +
from tc_formation.binary_classifications.data.patches_data_loader import PatchesDataLoader # noqa


def perform_prediction_on_patches(model, patches_ds) -> pd.DataFrame:
    results = []
    # nb_batches = len(patches_ds)
    
    for X, coords, paths in tqdm(iter(patches_ds)):
        nb_patches = X.shape[0]
        # Perform prediction on these patches.
        # print('before')
        pred = model.predict(X, verbose=False)
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


dataloader = PatchesTFRecordDataLoader()
test_path = 'data/patches_RCP45_size_30_stride_5_0h/data_all_variables_31_31.tfrecords'
test_patches_ds = dataloader.load_dataset(test_path, batch_size=128)
results_df = perform_prediction_on_patches(model, test_patches_ds)
results_df.to_csv('other_experiments/binary_classification_all_patches/future_projection/06_exp02_future_projection_RCP45.csv', index=False)
results_df.head()
# -

results_df[results_df['pred'] > 0.5].head()

len(results_df[results_df['pred'] >= 0.5]) / len(results_df)

# ## RCP85

test_path = 'data/patches_RCP85_size_30_stride_5_0h/data_all_variables_31_31.tfrecords'
test_patches_ds = dataloader.load_dataset(test_path, batch_size=128)
results_df = perform_prediction_on_patches(model, test_patches_ds)
results_df.to_csv('other_experiments/binary_classification_all_patches/future_projection/06_exp02_future_projection_RCP85.csv', index=False)
results_df.head()

results_df[results_df['pred'] > 0.5].head()

len(results_df[results_df['pred'] >= 0.5]) / len(results_df)
