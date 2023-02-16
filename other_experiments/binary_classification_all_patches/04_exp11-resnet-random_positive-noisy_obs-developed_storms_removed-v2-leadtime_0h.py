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

import pickle
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.binary_classifications.data.random_positive_patches_data_loader import RandomPositivePatchesDataLoader
from tc_formation.layers.sklearn_pca import SklearnPCALayer
from tc_formation.layers.residual_block import ResidualBlock
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from tqdm.auto import tqdm
import numpy as np
from sklearn.manifold import TSNE
# -


# In this version,
# I'll use the new developed storms removal datasets (v2).
#
# And, I will use Resnet-like model.

# +
dataloader = PatchesWithGenesisTFRecordDataLoader()
train_negative_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_variables_developed_storms_removed_v2_Train.tfrecords'
val_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_variables_developed_storms_removed_v2_Val.tfrecords'
test_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_variables_developed_storms_removed_v2_Test.tfrecords'

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
train_random_positive_path = 'data/ncep_WP_EP_0h_full_domain_developed_storms_removed_v2_Train.tfrecords'
random_positive_patch_dataloader = RandomPositivePatchesDataLoader(
    datashape=(41, 161, 136),
    domain_size=31)
train_random_positive_patches_ds = random_positive_patch_dataloader.load_dataset(train_random_positive_path)

# Merge positive and negative datasets to form our train dataset.
train_patches_ds = tf.data.Dataset.sample_from_datasets(
    [train_random_positive_patches_ds.repeat(1),
     train_fixed_patches_ds],
    weights=[0.5, 0.5],
    stop_on_empty_dataset=False)
train_patches_ds = train_patches_ds.batch(256)
# -

# ## Resnet-like Model
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
    # SklearnPCALayer(pca.components_, pca.explained_variance_),
    layers.GaussianNoise(1.),
], name='preprocessing')

# Now, we can define the model, similar to what we did in binary_classifications.

# +
class F1(tf.keras.metrics.Metric):
    def __init__(self, thresholds, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)

        self._thresholds = thresholds
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

    def get_config(self):
        return dict(thresholds=self._thresholds)


model = keras.Sequential([
    layers.Input(input_shape),
    preprocessing,
    layers.Conv2D(
        128, 7, strides=2, activation='relu', padding='SAME'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(128, name='block_1a'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(128, name='block_1b'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(256, stride1=2, name='block_2a'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(256, name='block_2b'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(512, stride1=2, name='block_3a'),
    layers.LayerNormalization(axis=-1),
    ResidualBlock(512, name='block_3b'),
    layers.LayerNormalization(axis=-1),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-3)),
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
    epochs=500,
    validation_data=val_patches_ds,
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            mode='max',
            verbose=1,
            patience=20,
            restore_best_weights=True),
    ]
)
# -

metrics = model.evaluate(test_patches_ds)
metrics

model.save(f'saved_models/binary_classification_all_patches/04_exp11_{metrics[-1]:.3f}')

# ## Feature Maps Visualization

# +
from sklearn.manifold import TSNE # noqa
import seaborn as sns # noqa
import pandas as pd # noqa
import matplotlib.pyplot as plt # noqa

feature_map = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name='flatten').output,
)

def obtain_embedding(patches_ds):
    embeddings = []
    labels = []
    preds = []

    for X, y in iter(patches_ds):
        X_emb = feature_map(X)
        y_pred = np.where(model(X).numpy() < 0.5, 0, 1)
        # print(y_pred[:5], y_pred.shape)

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


def perform_tsne(df):
    tsne = TSNE()
    embedding = np.asarray([x.tolist() for x in df['embedding'].tolist()])
    embedding_2 = tsne.fit_transform(embedding)

    df = df.copy()
    df['f1'] = embedding_2[:, 0]
    df['f2'] = embedding_2[:, 1]
    # df[['f1', 'f2']] = embedding_2[:, 0], embedding_2[:, 1]
    return df


# Now, we will apply the encoder to the validation set
# and visualize the embedding vectors with tSNE.
val_embedding_df = obtain_embedding(val_patches_ds)
df = perform_tsne(val_embedding_df)
# -

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
sns.scatterplot(df, x='f1', y='f2', hue='y', style='y', ax=axes[0])
sns.scatterplot(df, x='f1', y='f2', hue='y_pred', style='y_pred', ax=axes[1])

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

sns.scatterplot(
    df[(df['type'] == 'val') & (df['y'] == 1)], x='f1', y='f2', hue='y', style='y')
plt.title('Genesis cases in validation data')

sns.scatterplot(
    df[(df['type'] == 'val') & (df['y'] == 0)], x='f1', y='f2', hue='y', style='y')
plt.title('Non-genesis cases in validation data')

sns.scatterplot(
    df[(df['type'] == 'train') & (df['y'] == 1)], x='f1', y='f2', hue='y', style='y')
plt.title('Genesis cases in training data')

sns.scatterplot(
    df[(df['type'] == 'train') & (df['y'] == 0)], x='f1', y='f2', hue='y', style='y')
plt.title('Non-genesis cases in training data')
