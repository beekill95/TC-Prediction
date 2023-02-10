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
from vae import VAE

# %cd ../../..
# %load_ext autoreload
# %autoreload 2

from collections import Counter
import pickle
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.binary_classifications.data.random_positive_patches_data_loader import RandomPositivePatchesDataLoader
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
import numpy as np
# -

# # Variation AutoEncoder
#
# In this version,
# I'll use the new developed storms removal datasets (v2).

# +
dataloader = PatchesWithGenesisTFRecordDataLoader()
train_negative_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_developed_storms_removed_v2_Train.tfrecords'
val_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_developed_storms_removed_v2_Val.tfrecords'
test_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_developed_storms_removed_v2_Test.tfrecords'

train_fixed_patches_ds = dataloader.load_dataset(train_negative_path, batch_size=-1)
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256)
test_patches_ds = dataloader.load_dataset(test_path, batch_size=256)

input_shape = (31, 31, 136)
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


train_fixed_patches_ds = train_fixed_patches_ds.map(set_shape(input_shape, False))
val_patches_ds = val_patches_ds.map(set_shape(input_shape))
test_patches_ds = test_patches_ds.map(set_shape(input_shape))

# +
# Load positive patch dataset.
train_random_positive_path = 'data/ncep_WP_EP_6h_full_domain_developed_storms_removed_v2_Train.tfrecords'
random_positive_patch_dataloader = RandomPositivePatchesDataLoader(
    datashape=(41, 161, 136),
    domain_size=31)
train_random_positive_patches_ds = random_positive_patch_dataloader.load_dataset(train_random_positive_path)

# Merge positive and negative datasets to form our train dataset.
train_patches_ds = tf.data.Dataset.sample_from_datasets(
    [train_random_positive_patches_ds, train_fixed_patches_ds],
    weights=[0.5, 0.5],
    stop_on_empty_dataset=False).map(lambda X, y: X)
train_patches_ds = train_patches_ds.batch(256)
# -

# ## Variational Autoencoder Model

# +
def load_pickle(path: str):
    with open(path, 'rb') as inpath:
        obj = pickle.load(inpath)
        return obj

scaler = load_pickle('scalerdeveloped_storms_removed_v2.pkl')
# pca = load_pickle('pcadeveloped_storms_removed_v2.pkl')

preprocessing = keras.Sequential([
    layers.Normalization(mean=scaler.mean_, variance=scaler.var_),
    # layers.GaussianNoise(1.),
    # SklearnPCALayer(pca.components_),
], name='preprocessing')


def EncoderModel(input_shape, preprocessing, latent_dim: int):
    inputs = layers.Input(input_shape)
    x = keras.Sequential([
        # preprocessing,
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
        layers.LayerNormalization(axis=-1),
        layers.Dropout(0.5),
        ])(inputs)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    return keras.Model(inputs, [z_mean, z_log_var], name='encoder')


def DecoderModel(latent_dim: int, scaler):
    return keras.Sequential([
        layers.Input((latent_dim,)),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)),
        layers.LayerNormalization(axis=-1),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)),
        layers.Reshape((1, 1, 512)),
        layers.LayerNormalization(axis=-1),
        layers.Conv2DTranspose(
            512, 3,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(1e-3)),
        layers.LayerNormalization(-1),
        layers.Conv2DTranspose(
            256, 3,
            strides=2,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(1e-3)),
        layers.LayerNormalization(-1),
        layers.Conv2DTranspose(
            128, 3,
            strides=2,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(1e-3)),
        layers.LayerNormalization(-1),
        layers.Conv2DTranspose(
            136, 3,
            strides=2,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(1e-3)),
        # layers.Lambda(
        #     lambda x: (x * tf.sqrt(tf.convert_to_tensor(scaler.var_, dtype=tf.float32))
        #                + tf.convert_to_tensor(scaler.mean_, dtype=tf.float32))),
        # layers.Normalization(mean=scaler.mean_, variance=scaler.var_, invert=True),
    ], name='Decoder')


latent_dim = 512
encoder_model = EncoderModel(
    input_shape,
    preprocessing,
    latent_dim)
encoder_model.summary()
# -

decoder_model = DecoderModel(latent_dim, scaler)
decoder_model.summary()

# + tags=[]
std = tf.convert_to_tensor(np.sqrt(scaler.var_), dtype=tf.float64)
mean = tf.convert_to_tensor(scaler.mean_, dtype=tf.float64)
train_patches_ds = train_patches_ds.map(lambda X: (X - mean) / std)

vae_model = VAE(encoder_model, decoder_model)
vae_model.compile(optimizer=keras.optimizers.Adam())
vae_model.fit(train_patches_ds, epochs=500)
# -

encoder_model.save(
    f'saved_models/vae_encoder_02_exp01')

# Now, we will apply the encoder to the validation set
# and visualize the embedding vectors with tSNE.
val_embedding = []
val_label = []
for X, y in iter(val_patches_ds):
    X = (X - mean) / std
    X_emb, X_std = encoder_model(X)
    val_embedding.append(np.concatenate([X_emb, X_std], axis=1))
    val_label.append(y)

val_embedding = np.concatenate(val_embedding, axis=0)
val_label = np.concatenate(val_label, axis=0)

# +
from sklearn.manifold import TSNE # noqa
import pandas as pd # noqa
import seaborn as sns # noqa
import matplotlib.pyplot as plt # noqa

tsne = TSNE()
val_embedding_2 = tsne.fit_transform(val_embedding)
df = pd.DataFrame({
    'f1': val_embedding_2[:, 0],
    'f2': val_embedding_2[:, 1],
    'y': val_label[:, 0],
})
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(df, x='f1', y='f2', hue='y', style='y')
