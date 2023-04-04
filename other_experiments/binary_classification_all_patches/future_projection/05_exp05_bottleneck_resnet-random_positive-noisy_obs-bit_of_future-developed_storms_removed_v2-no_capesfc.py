# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %cd ../../..
# %load_ext autoreload
# %autoreload 2

import pickle
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.binary_classifications.data.random_positive_patches_data_loader import RandomPositivePatchesDataLoader
from tc_formation.layers.sklearn_pca import SklearnPCALayer
from tc_formation.layers.residual_block import BottleneckResidualBlock
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from tqdm.auto import tqdm
import numpy as np
# -


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# In this version,
# I'll use the new developed storms removal datasets (v2).

# +
dataloader = PatchesWithGenesisTFRecordDataLoader()
train_fixed_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_varibles_no_capesfc_developed_storms_removed_v2_Train.tfrecords'
val_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_varibles_no_capesfc_developed_storms_removed_v2_Val.tfrecords'
test_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_varibles_no_capesfc_developed_storms_removed_v2_Test.tfrecords'

train_rcp45_fixed_path = 'data/WPAC_RCP45_all_patches_all_variables_till_20350101_leadtime_0h.tfrecords'

train_fixed_patches_ds = dataloader.load_dataset(train_fixed_path, batch_size=-1)
train_RCP45_fixed_patches_ds = dataloader.load_dataset(train_rcp45_fixed_path, batch_size=-1)
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256)
test_patches_ds = dataloader.load_dataset(test_path, batch_size=256)

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


train_fixed_patches_ds = train_fixed_patches_ds.map(set_shape(input_shape, False))
train_RCP45_fixed_patches_ds = train_RCP45_fixed_patches_ds.map(set_shape(input_shape, False))
val_patches_ds = val_patches_ds.map(set_shape(input_shape))
test_patches_ds = test_patches_ds.map(set_shape(input_shape))

# +
# Load positive patch dataset.
train_random_positive_path = 'data/ncep_WP_EP_0h_full_domain_developed_storms_removed_v2_no_capesfc_Train.tfrecords'
train_random_positive_RCP45_path = 'data/WPAC_RCP45_full_domain_all_variables_till_20350101_leadtime_0h.tfrecords'
random_positive_patch_dataloader = RandomPositivePatchesDataLoader(
    datashape=(41, 161, 135),
    domain_size=31)
train_random_positive_patches_ds = random_positive_patch_dataloader.load_dataset(train_random_positive_path)
train_random_positive_RCP45_patches_ds = random_positive_patch_dataloader.load_dataset(train_random_positive_RCP45_path)

# Merge positive and negative datasets to form our train dataset.
train_patches_ds = tf.data.Dataset.sample_from_datasets(
    [train_random_positive_patches_ds,
     train_fixed_patches_ds,
     train_RCP45_fixed_patches_ds,
     train_random_positive_RCP45_patches_ds,
    ],
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

scaler = load_pickle('scaler_developed_storms_removed_v2_no_capesfc.pkl')
# pca = load_pickle('pca_developed_storms_removed_v2_no_capesfc.pkl')

preprocessing = keras.Sequential([
    layers.Normalization(mean=scaler.mean_, variance=scaler.var_),
    layers.GaussianNoise(1.),
    # SklearnPCALayer(pca.components_),
], name='preprocessing')
# -

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
    BottleneckResidualBlock(128, name='block_1a'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(128, name='block_1b'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(128, name='block_1c'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(256, stride1=2, name='block_2a'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(256, name='block_2b'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(256, name='block_2c'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(512, stride1=2, name='block_3a'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(512, name='block_3b'),
    layers.LayerNormalization(axis=-1),
    BottleneckResidualBlock(512, name='block_3c'),
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
# -

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
            patience=50,
            restore_best_weights=True),
    ]
)

metrics = model.evaluate(test_patches_ds)
metrics

model.save(f'saved_models/binary_classification_all_patches/future_projection/05_exp05_f1_{metrics[-1]:.3f}')

# ### Model Performance at Different Threshold Values

# +
from sklearn.metrics import precision_score, recall_score, f1_score # noqa
import numpy as np # noqa


yy_pred = []
yy_true = []
for X, y in tqdm(iter(test_patches_ds)):
    y_pred = model.predict(X, verbose=0)
    yy_pred.append(y_pred)
    yy_true.append(y)

yy_pred = np.concatenate(yy_pred, axis=0).flatten()
yy_true = np.concatenate(yy_true, axis=0).flatten()

# +
import pandas as pd # noqa


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
