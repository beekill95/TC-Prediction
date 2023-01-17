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

# +
from __future__ import annotations

from datetime import datetime
from tc_formation import plot
from tc_formation.data.loaders import tc_occurence_time_range as time_range
from tc_formation.models import resnet
import tc_formation.tf_metrics as tfm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
# -

# # ResNet with Time Range Output

# Specify the configurations for this experiment.

exp_name = 'time_range_resnet_24h'
runtime = datetime.now().strftime('%Y_%m_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_time_range_24h_with_0h_no_genesis_class.csv'
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
data_shape = (41, 161, 13)
patch_size = 16
leadtime = 12

# ## Data Loading

# +
loader = time_range.TropicalCycloneOccurenceTimeRangeDataLoader(
        data_shape=data_shape, subset=subset)
training_ds = loader.load_dataset(train_path, shuffle=True)
validation_ds = loader.load_dataset(val_path)

X_train = training_ds.map(lambda X, _: X)
# -

# ## Model

inputs = layers.Input(data_shape)
preprocessing = keras.Sequential([
    layers.Normalization(axis=-1),
], name='preprocessing')
preprocessing.layers[0].adapt(X_train)

model = keras.Sequential([
    layers.Input(data_shape),
    preprocessing,
    # layers.Conv2D(256, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-4)),
    layers.Conv2D(64, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-4)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    # layers.Conv2D(512, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-4)),
    layers.Conv2D(128, 3, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-4)),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, kernel_regularizer=keras.regularizers.L2(1e-4), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, kernel_regularizer=keras.regularizers.L2(1e-4)),
])
model.build()
model.summary()


# +
def weighted_sigmoid_cross_entropy_loss(from_logits: bool = False):
    binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def loss(y_true, y_pred):
        loss = 0.
        for i in range(y_true.shape[1]):
            # Manually scale the sample weight.
            sample_weight = tf.where(y_true[:, i] == 1, 40., 1.)[:, None]
            # tf.print(y_true[:5, i])
            loss += binary_loss(y_true[:, i], y_pred[:, i], sample_weight)
            # tf.print(sample_weight)
        return loss

    return loss


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    # loss=tf.compat.v1.losses.sigmoid_cross_entropy,
    loss=weighted_sigmoid_cross_entropy_loss(from_logits=True),
    # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        tfm.RecallScore(from_logits=True),
        tfm.PrecisionScore(from_logits=True),
        tfm.F1Score(num_classes=5, from_logits=True, threshold=0.5),
    ])
# -

# ## Training & Testing

# +
epochs = 500
first_stage_history = model.fit(
    training_ds,
    epochs=epochs,
    validation_data=validation_ds,
    # class_weight={1: 20., 0: 1.},
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

# plot.plot_training_history(first_stage_history, "First stage training")
# -

testing_ds = loader.load_dataset(test_path)
model.evaluate(testing_ds)
