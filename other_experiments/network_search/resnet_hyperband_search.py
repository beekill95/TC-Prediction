# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import sys  # noqa
sys.path.append('..')  # noqa

import models.resnet as resnet
import keras_tuner as kt
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import tf_metrics as tfm
import data
import plot
import tensorflow as tf
import tensorflow.keras as keras
# -

# # Hyperband Search with Resnet


# + tags=["parameters"]
project_start_time = '2021Nov2_1518'
# -

# First, load training data and validation data.

data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
train_path = f'{data_path}_train'
val_path = f'{data_path}_val'
test_path = f'{data_path}_test'
data_shape = (41, 181, 135)

full_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
)
downsampled_training = data.load_data(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    negative_samples_ratio=1)
validation = data.load_data(val_path, data_shape=data_shape)

normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
for X, y in iter(full_training):
    normalizer.adapt(X)


def normalizer(x): return x


# +
def normalize_data(x, y):
    return normalizer(x), y


full_training = full_training.map(normalize_data)
downsampled_training = downsampled_training.map(normalize_data)
validation = validation.map(normalize_data)
# -

# This is the model we will be tuning.


def build_resnet_50_model(hp):
    filters_size = [32, 64, 128, 256, 512, 1024]
    nb_blocks = [2, 3, 4, 5, 6, 7, 8]

    def stack_fn(x):
        x = resnet._stack1(
            x,
            hp.Choice('conv2_filters', filters_size),
            hp.Choice('conv2_nb_blocks', nb_blocks),
            #3,
            stride1=1,
            name='conv2')
        x = resnet._stack1(
            x,
            hp.Choice('conv3_filters', filters_size),
            hp.Choice('conv3_nb_blocks', nb_blocks),
            #4,
            name='conv3')
        x = resnet._stack1(
            x,
            hp.Choice('conv4_filters', filters_size),
            hp.Choice('conv4_nb_blocks', nb_blocks),
            #6,
            name='conv4')
        return resnet._stack1(
            x,
            hp.Choice('conv5_filters', filters_size),
            hp.Choice('conv5_nb_blocks', nb_blocks),
            #3,
            name='conv5')

    model = resnet._ResNet(stack_fn,
                           False,
                           True,
                           model_name='random_search_resnet18',
                           include_top=True,
                           input_tensor=None,
                           input_shape=data_shape,
                           pooling=None,
                           classes=1,
                           classifier_activation=None)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 1e-3, 1e-4])),
        loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
        metrics=[
            'binary_accuracy',
            tfm.RecallScore(from_logits=True),
            tfm.PrecisionScore(from_logits=True),
            tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
        ]
    )

    return model

# Begin our quest for the best parameters.


tuner = kt.Hyperband(
    build_resnet_50_model,
    objective=kt.Objective('val_f1_score', direction='max'),
    max_epochs=80,
    directory='hyperband_resnet50',
    project_name=f'depth_blocks_learning_rate_{project_start_time}'
)
tuner.search(
    downsampled_training,
    epochs=50,
    validation_data=validation,
    class_weight={1: 1., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ]
)

tuner.results_summary()

# After we got the result, first,
# test on the testing to see what kind of performance we're getting.

# +
model = tuner.get_best_models()[0]

testing = data.load_data(test_path, data_shape=data_shape)
testing = testing.map(normalize_data)
model.evaluate(testing)
# -

# Finally, train on full dataset.

# +
epochs = 50
second_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    class_weight={1: 10., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            patience=10,
            restore_best_weights=True),
    ])


plot.plot_training_history(second_stage_history, "")
# -

# And get the result.
model.evaluate(testing)
