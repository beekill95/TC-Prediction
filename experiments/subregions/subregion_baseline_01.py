# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: tc_updated_2
#     language: python
#     name: tc_updated_2
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

from datetime import datetime
import tc_formation.data.subregions as subregions
import tc_formation.data.subregions.data_loader
from tc_formation.models.subregion_baseline import SubregionBaseline
import tc_formation.tf_metrics as tfm
import tensorflow as tf
import tensorflow.keras as keras

# # Sub Region Baseline

# ## Configurations

exp_name = 'subregion_baseline_vortex_removal_12h'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/tc_ibtracs_12h_WP_EP_v4.csv'
# data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')
subset = dict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
)
subregion_size = (20, 20) # in degree
subregion_stride = 5 # in degree
data_shape = (41, 161, 13)

# ## Dataset Loading

# +
data_loader = subregions.data_loader.SubRegionsTropicalCycloneDataLoader(
    data_shape=data_shape,
    subset=subset,
    subregion_size=subregion_size,
    subregion_stride=subregion_stride,
)
training = data_loader.load_dataset_wip(
    train_path,
    batch_size=512,
    shuffle=False,
    negative_subregions_ratio=5,
    caching=False,
    nonTCRatio=3,
    other_happening_tc_ratio=None,
)
validation = data_loader.load_dataset_wip(
    val_path,
    batch_size=256,
    shuffle=False,
    negative_subregions_ratio=None,
)

def log_percentage(X, y):
    batch_size = tf.cast(tf.shape(y)[0], dtype=tf.float32)
    nb_true = tf.reduce_sum(y)
    tf.print('\n==== % of true', nb_true, nb_true / batch_size)
    return X, y

training = training.map(log_percentage)
# -

# ## Model

model = SubregionBaseline(
    input_shape=subregion_size + data_shape[-1:],
    classes=1,
    output_activation=None,
    name='baseline',
)
model.summary()

# ### Model Training

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.CustomF1Score(from_logits=True),
    ]
)

epochs = 150
first_stage_history = model.fit(
    training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 5., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=20,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f'outputs/{exp_name}_{runtime}',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}',
        ),
    ]
)
