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

from collections import Counter
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.layers.sklearn_pca import SklearnPCALayer
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
import numpy as np


# +
dataloader = PatchesWithGenesisTFRecordDataLoader()
train_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_remove_outside_grouped_Train.tfrecords'
val_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_remove_outside_grouped_Val.tfrecords'
test_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_remove_outside_grouped_Test.tfrecords'

train_patches_ds = dataloader.load_dataset(train_path, batch_size=256, shuffle=True)
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256)
test_patches_ds = dataloader.load_dataset(test_path, batch_size=256)

input_shape = (31, 31, 136)
def set_shape(X, y):
    X.set_shape((None, ) + input_shape)
    y.set_shape((None, 1))
    return X, y

train_patches_ds = train_patches_ds.map(set_shape)
val_patches_ds = val_patches_ds.map(set_shape)
test_patches_ds = test_patches_ds.map(set_shape)

# +
def count_positive_samples(ds):
    counter = Counter()
    for _, y in iter(ds):
        counter.update(y.numpy().flatten())

    return counter

print(f'Training set statisitics: {count_positive_samples(train_patches_ds)}')
print(f'Validation set statisitics: {count_positive_samples(val_patches_ds)}')
print(f'Test set statisitics: {count_positive_samples(test_patches_ds)}')
# -

# ## Model
#
# Load preprocessing pipeline.

# +
def load_pickle(path: str):
    with open(path, 'rb') as inpath:
        obj = pickle.load(inpath)
        return obj

scaler = load_pickle('scaler.pkl')
pca = load_pickle('pca.pkl')

preprocessing = keras.Sequential([
    layers.Normalization(mean=scaler.mean_, variance=scaler.var_),
    layers.GaussianNoise(1.),
    SklearnPCALayer(pca.components_),
], name='preprocessing')
# -

# Now, we can define the model, similar to what we did in binary_classifications.

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


model = keras.Sequential([
    layers.Input(input_shape),
    preprocessing,
    layers.Conv2D(
        # was 64, and achieved 0.31 recall on test data. )but with only 2 conv2d)
        # with 128, achieved .48 recall on test data. )but with only 2 conv2d)
        128, 3,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.LayerNormalization(axis=-1),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(
        256, 3,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.LayerNormalization(axis=-1),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(
        512, 3,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.LayerNormalization(axis=-1),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.Dropout(0.5),
    layers.Dense(1, kernel_regularizer=keras.regularizers.L2(1e-3)),
    layers.Activation('sigmoid'),
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        'binary_accuracy',
        tf.keras.metrics.Precision(thresholds=0.5),
        tf.keras.metrics.Recall(thresholds=0.5),
        F1(thresholds=0.5),
    ]
)
model.summary()

# + tags=[]
model.fit(
    train_patches_ds,
    epochs=100,
    validation_data=val_patches_ds,
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            verbose=1,
            patience=50,
            restore_best_weights=True),
    ]
)
# -

model.evaluate(test_patches_ds)
