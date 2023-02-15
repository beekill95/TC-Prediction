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
from twin_nn import TwinNN

# %cd ../../..
# %load_ext autoreload
# %autoreload 2

import pickle
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.binary_classifications.data.random_positive_patches_data_loader import RandomPositivePatchesDataLoader
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# -


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
    [train_random_positive_patches_ds.repeat(10),
     train_fixed_patches_ds],
    weights=[0.5, 0.5],
    stop_on_empty_dataset=False)
train_patches_ds = train_patches_ds.batch(256)
# -

# ## Model
#
# Load preprocessing pipeline.

# +
def load_pickle(path: str):
    with open(path, 'rb') as inpath:
        obj = pickle.load(inpath)
        return obj

scaler = load_pickle('scalerdeveloped_storms_removed_v2.pkl')
# pca = load_pickle('pcadeveloped_storms_removed_v2.pkl')

preprocessing = keras.Sequential([
    layers.Normalization(mean=scaler.mean_, variance=scaler.var_),
    layers.GaussianNoise(1.),
    # SklearnPCALayer(pca.components_),
], name='preprocessing')
# -

# Now, we can define the model, similar to what we did in binary_classifications.

# + tags=[]
latent_dim = 1024
feature_extractor = keras.Sequential([
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
    layers.Dense(
        latent_dim,
        activation='relu',
        kernel_regularizer=keras.regularizers.L2(1e-3),
    ),
    layers.LayerNormalization(axis=-1, name='feature_out'),
])
pos_head = keras.Sequential([
    layers.Input(latent_dim),
    layers.Dense(1024, kernel_regularizer=keras.regularizers.L2(1e-3), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, kernel_regularizer=keras.regularizers.L2(1e-3), name='pos_out'),
])
neg_head = keras.Sequential([
    layers.Input(latent_dim),
    layers.Dense(1024, kernel_regularizer=keras.regularizers.L2(1e-3), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, kernel_regularizer=keras.regularizers.L2(1e-3), name='neg_out'),
])

twin_nn = TwinNN(
    base=feature_extractor,
    pos_head=pos_head,
    pos_head_weight=pos_head.get_layer(name='pos_out').weights[0],
    neg_head=neg_head,
    neg_head_weight=neg_head.get_layer(name='neg_out').weights[0],
)
twin_nn.compile(optimizer=keras.optimizers.Adam())
twin_nn.fit(
    train_patches_ds,
    validation_data=val_patches_ds,
    epochs=100,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            verbose=1,
            patience=20,
            restore_best_weights=True),
    ])
# -

twin_nn.evaluate(test_patches_ds)

# +
import numpy as np # noqa
import pandas as pd # noqa


def obtain_embedding(patches_ds):
    embeddings = []
    labels = []
    preds = []

    for X, y in iter(patches_ds):
        X_emb = feature_extractor(X)
        y_pred = twin_nn(X)

        embeddings.append(X_emb)
        labels.append(y)
        preds.append(y_pred)

    embeddings, labels, preds = tuple(
        np.concatenate(x, axis=0)
        for x in [embeddings, labels, preds])
    return pd.DataFrame({
        'embedding': list(embeddings),
        'y': labels[:, 0],
        'y_pred': preds[:, 0],
    })


# Now, we will apply the encoder to the validation set
# and visualize the embedding vectors with tSNE.
val_embedding_df = obtain_embedding(val_patches_ds)

# +
from sklearn.metrics import f1_score, precision_score, recall_score # noqa

# We can also check the f1 score, precision, and recall on the validation set.
val_f1, val_precision, val_recall = tuple(
    fn(val_embedding_df['y'], val_embedding_df['y_pred'])
    for fn in [f1_score, precision_score, recall_score])
print(f'{val_f1=}, {val_precision=}, {val_recall=}')

# +
from sklearn.manifold import TSNE # noqa
import seaborn as sns # noqa
import matplotlib.pyplot as plt # noqa


def perform_tsne(df):
    tsne = TSNE()
    embedding = np.asarray([x.tolist() for x in df['embedding'].tolist()])
    embedding_2 = tsne.fit_transform(embedding)

    df = df.copy()
    df['f1'] = embedding_2[:, 0]
    df['f2'] = embedding_2[:, 1]
    # df[['f1', 'f2']] = embedding_2[:, 0], embedding_2[:, 1]
    return df


fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
df = perform_tsne(val_embedding_df)
sns.scatterplot(df, x='f1', y='f2', hue='y', style='y', ax=axes[0])
sns.scatterplot(df, x='f1', y='f2', hue='y_pred', style='y_pred', ax=axes[1])
# -

# Now, we can check the feature of both training and validation.
train_embedding_df = obtain_embedding(train_patches_ds)
train_embedding_df['type'] = 'train'
val_embedding_df['type'] = 'val'
embedding_df = pd.concat([train_embedding_df, val_embedding_df])
df = perform_tsne(embedding_df)

# df['type y'] = df.apply(lambda r: f"{r['type']}-{r['y']}", axis=1)
# df['type y pred'] = df.apply(lambda r: f"{r['type']}-{r['y_pred']}", axis=1)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
sns.scatterplot(
    df[df['type'] == 'train'], x='f1', y='f2', hue='y', style='y', ax=axes[0, 0])
sns.scatterplot(
    df[df['type'] == 'train'], x='f1', y='f2', hue='y_pred', style='y_pred', ax=axes[0, 1])
sns.scatterplot(
    df[df['type'] == 'val'], x='f1', y='f2', hue='y', style='y', ax=axes[1, 0])
sns.scatterplot(
    df[df['type'] == 'val'], x='f1', y='f2', hue='y_pred', style='y_pred', ax=axes[1, 1])
