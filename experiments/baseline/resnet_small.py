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
import tc_formation.models.layers
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tc_formation.data.loaders.tc_occurence import TimeSeriesTropicalCycloneOccurenceDataLoader, TropicalCycloneOccurenceDataLoader
import tensorflow_addons as tfa
from datetime import datetime

# # Use Small ResNet

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

exp_name = 'baseline_resnet_6'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb'
#data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
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
data_shape = (41, 161, 26)

# + tags=[]
model = resnet.ResNet6(
    input_shape=data_shape,
    include_top=True,
    classes=1,
    classifier_activation=None,)
model.summary()
# -

# Build the model using BinaryCrossentropy loss

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.CustomF1Score(from_logits=True),
    ]
)

# Load our training and validation data.

# full_training = data.load_data_v1(
#     train_path,
#     data_shape=data_shape,
#     batch_size=64,
#     shuffle=True,
#     subset=subset,
#     group_same_observations=False,
# )
loader = TimeSeriesTropicalCycloneOccurenceDataLoader(data_shape=(*data_shape[:2], 13), subset=subset, previous_hours=[6])
full_training = loader.load_dataset(data_path=train_path, batch_size=128)
validation = loader.load_dataset(data_path=val_path)
# downsampled_training = data.load_data(
#     train_path,
#     data_shape=data_shape,
#     batch_size=64,
#     shuffle=True,
#     subset=subset,
#     negative_samples_ratio=1)
# validation = data.load_data_v1(
#     val_path,
#     data_shape=data_shape,
#     subset=subset,
#     group_same_observations=True,
# )

# +
def concat_along_2nd_dim(x, y):
    return tf.concat([x[:, 0], x[:, 1]], axis=-1), y

normalizer = preprocessing.Normalization()
normalizer.adapt(full_training.map(concat_along_2nd_dim).map(lambda x, _: x))


def normalize_data(x, y):
    return normalizer(x), y

full_training = full_training.map(concat_along_2nd_dim).map(normalize_data)
# downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(concat_along_2nd_dim).map(normalize_data)
# -

# # First stage
#
# train the model on the down-sampled data.

# +
epochs = 500
first_stage_history = model.fit(
    # downsampled_training,
    full_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 10., 0: 1.},
    shuffle=True,
    callbacks=[
        # keras.callbacks.EarlyStopping(
        #     monitor='val_f1_score',
        #     mode='max',
        #     verbose=1,
        #     patience=20,
        #     restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp",
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

testing = data.load_data_v1(
    test_path,
    data_shape=data_shape,
    subset=subset,
    group_same_observations=True,
)
testing = testing.map(concat_along_2nd_dim).map(normalize_data)
model.evaluate(
    testing,
    callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
    ])

# Obtain performance value at different thresholds.

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in thresholds:
    model.compile(
        optimizer='adam',
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
        metrics=[
            'binary_accuracy',
            tfm.RecallScore(thresholds=t, from_logits=True),
            tfm.PrecisionScore(thresholds=t, from_logits=True),
            tfm.CustomF1Score(thresholds=t, from_logits=True),
        ]
    )
    print(f'=== Threshold {t} ===')
    model.evaluate(testing)

# # Second stage
#
# train the model on full dataset.

# +
# second_stage_history = model.fit(
#     full_training,
#     epochs=epochs,
#     validation_data=validation,
#     class_weight={1: 10., 0: 1.},
#     shuffle=True,
#     callbacks=[
#         keras.callbacks.EarlyStopping(
#             monitor='val_f1_score',
#             mode='max',
#             verbose=1,
#             patience=20,
#             restore_best_weights=True),
#         keras.callbacks.ModelCheckpoint(
#             filepath=f"outputs/{exp_name}_{runtime}_2nd_ckp",
#             monitor='val_f1_score',
#             mode='max',
#             save_best_only=True,
#         ),
#     ])


# plot.plot_training_history(second_stage_history, "Second stage training")
# -

# After the model is trained, we will test it on test data.

# model.evaluate(testing)
