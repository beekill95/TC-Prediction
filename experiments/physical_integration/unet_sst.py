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

from collections import OrderedDict
from datetime import datetime
from tc_formation.models import unet
from tc_formation import tf_metrics as tfm
from tc_formation.metrics.bb import BBoxesIoUMetric
from tc_formation.losses import physical_consistent_losses as pcl
import tc_formation.data.time_series as ts_data
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

# # Predict TC Formation using Grid Probability with Physical Knowledge

exp_name = 'tc_grid_prob_unet_physical_integration'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/ncep_WP_EP_new/TODO'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')
subset = OrderedDict(
    absvprs=[900, 750],
    rhprs=[750],
    tmpprs=[900, 500],
    hgtprs=[500],
    vvelprs=[500],
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
    capesfc=True,
    tmpsfc=True,
)
data_shape = (41, 161, 13)

# ## Unet Model

# +
input_layer = keras.Input(data_shape)
normalization_layer = layers.Normalization()
model = unet.Unet(
    input_tensor=normalization_layer(input_layer),
    model_name='unet',
    classifier_activation='sigmoid',
    output_classes=1,
    decoder_shortcut_mode='add',
    filters_block=[64, 128])

outputs = model.outputs

model.summary()
# -

# ## Data Loading

tc_avg_radius_lat_deg = 3
data_loader = ts_data.TropicalCycloneWithGridProbabilityDataLoader(
    data_shape=data_shape,
    tc_avg_radius_lat_deg=tc_avg_radius_lat_deg,
    subset=subset,
    softmax_output=False,
    smooth_gt=True,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=128,
    leadtimes=[12],
    shuffle=True)
validation = data_loader.load_dataset(
    val_path,
    leadtimes=[12],
    batch_size=128)

# ## Training

# Adapt normalization layer.

features = training.map(lambda feature, _: feature)
normalization_layer.adapt(features)

# +
from functools import reduce # noqa


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def multiple_losses(loss_fn: list, loss_weights: list | None):
    if loss_weights is None:
        loss_weights = [1.] * len(loss_fn)
    else:
        assert len(loss_fn) == len(loss_weights)
    
    def _combined_loss(y_true, y_pred):
        return reduce(
            lambda acc, cur: acc + cur[1] * cur[0](y_true, y_pred),
            zip(loss_fn, loss_weights), 0.0)
    
    return _combined_loss


model.compile(
    optimizer='adam',
    loss=multiple_losses([
        dice_loss,
        pcl.sst_loss(input_layer[:, :, :, 12]),
    ])
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # loss=combine_loss_funcs(hard_negative_mined_sigmoid_focal_loss, dice_loss),
    # loss=dice_loss,
    # loss=tf.losses.KLDivergence(),
    # loss=hard_negative_mined_sigmoid_focal_loss,
    # loss=hard_negative_mined_binary_crossentropy_loss,
    metrics=[
        #'binary_accuracy',
        #keras.metrics.Recall(name='recall', class_id=1 if use_softmax else None),
        #keras.metrics.Precision(name='precision', class_id=1 if use_softmax else None),
        #tfm.CustomF1Score(name='f1', class_id=1 if use_softmax else None),
        #BBoxesIoUMetric(name='IoU', iou_threshold=0.2),
        #tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        #tfm.PrecisionScore(from_logits=True),
        #tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ])

epochs = 150
model.fit(
    training,
    epochs=epochs,
    validation_data=validation,
    shuffle=True,
    callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=50,
            restore_best_weights=True
        ),
    ],
)
