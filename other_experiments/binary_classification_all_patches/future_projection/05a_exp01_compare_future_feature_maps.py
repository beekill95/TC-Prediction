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
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
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


path = 'saved_models/random_positive_no_pca_developed_storms_removed_v2_no_capesfc_leadtime_0h_f1_0.654'
model = tf.keras.models.load_model(path, compile=False)
model.compile(
    optimizer='adam',
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

# # Compare Feature Maps of Baseline and Future RCP45
# 
# First, load baseline data, which is NCEP/FNL dataset.

# +
dataloader = PatchesWithGenesisTFRecordDataLoader()
val_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_varibles_no_capesfc_developed_storms_removed_v2_Val.tfrecords'
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256)

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


val_patches_ds = val_patches_ds.map(set_shape(input_shape))
# -

# Now, we can load the RCP45 future data.

dataloader = PatchesTFRecordDataLoader()
outdir = 'other_experiments/binary_classification_all_patches/future_projection'
test_path = 'data/patches_RCP45_size_30_stride_5_0h/data_all_variables_31_31.tfrecords'
test_patches_ds = (dataloader
    .load_dataset(test_path, batch_size=128)
    .map(set_shape(input_shape)))

# +
from sklearn.manifold import TSNE # noqa
import seaborn as sns # noqa
import pandas as pd # noqa
import matplotlib.pyplot as plt # noqa

feature_map = tf.keras.Model(
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


