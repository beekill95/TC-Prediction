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

# %cd ../..

from tc_formation import plot
from tc_formation.data import data
from tc_formation.models.vision_transformer import ViT
from tc_formation.models.patches_layer import Patches
from tc_formation.data.loaders.tc_occurence import TropicalCycloneOccurenceDataLoader
import tc_formation.tf_metrics as tfm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime

# # Test for Single Leadtime

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

exp_name = 'baseline_test_12h'
runtime = datetime.now().strftime('%Y_%m_%d_%H_%M')
# data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v3.csv'
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/tc_ibtracs_12h_WP_EP_v3.csv.updated'
train_path = data_path.replace('.csv.updated', '_train.csv.updated')
val_path = data_path.replace('.csv.updated', '_val.csv.updated')
test_path = data_path.replace('.csv.updated', '_test.csv.updated')
subset = dict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
)
data_shape = (41, 161, 13)
patch_size = 16
# subset = dict(
#     hgtprs=[700, 500, 250],
#     ugrdprs=[700, 500, 250],
#     vgrdprs=[700, 500, 250],
#     capesfc=None,
#     absvprs=None,
#     rhprs=None,
#     tmpprs=None,
#     vvelprs=None,
#     tmpsfc=None,
#     slp=None,
# )
# data_shape = (41, 161, 9)
leadtime = 12

# + tags=[]
# input_tensor = layers.Input(data_shape)
# input_tensor = Patches(patch_size=16, flatten=True)(input_tensor)
# model = ViT(
#     input_tensor=input_tensor,
#     include_top=True,
#     sequence_length=33,
#     N=1,
#     attention_heads=1,
#     model_dim=64,
#     classes=1,
#     name='Vit_12h')
# -

# Build the model using BinaryCrossentropy loss

# Load our training and validation data.

loader = TropicalCycloneOccurenceDataLoader(data_shape=data_shape, subset=subset)
full_training = loader.load_dataset(data_path=train_path, batch_size=128, shuffle=True)
validation = loader.load_dataset(data_path=val_path)

normalizer = preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)
normalizer


# +
def normalize_data(x, y):
    return normalizer(x), y

full_training = full_training.map(normalize_data)
# downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(normalize_data)
# -

# ## Model

# +
# tf.keras.backend.clear_session()
model = keras.Sequential([
    layers.Input(data_shape),
    layers.Conv2D(256, 3, activation='relu'),
    # layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(256, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(1),
])
model.build()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ]
)
# -

# # First stage
#
# train the model on the down-sampled data.

# +
epochs = 500
first_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 20., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            # monitor='val_f1_score',
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=50,
            restore_best_weights=True),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=f"outputs/{exp_name}_{runtime}_1st_ckp",
        #     monitor='val_f1_score',
        #     mode='max',
        #     save_best_only=True,
        # ),
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
            histogram_freq=1,
        ),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data_v1(
    test_path,
    data_shape=data_shape,
    subset=subset,
    leadtime=leadtime,
    group_same_observations=True,
)
testing = testing.map(normalize_data)
print(f'\n**** LEAD TIME: {leadtime}')
model.evaluate(testing)

