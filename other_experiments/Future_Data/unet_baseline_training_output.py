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

# + papermill={"duration": 0.132369, "end_time": "2022-09-13T14:33:45.472510", "exception": false, "start_time": "2022-09-13T14:33:45.340141", "status": "completed"} tags=[]
# %cd ../..
# %load_ext autoreload
# %autoreload 2

# + papermill={"duration": 57.860586, "end_time": "2022-09-13T14:34:43.336690", "exception": false, "start_time": "2022-09-13T14:33:45.476104", "status": "completed"} tags=[]
from datetime import datetime
from tc_formation.models import unet
from tc_formation import tf_metrics as tfm
import tc_formation.metrics.bb as bb
import tc_formation.data.time_series as ts_data
from tc_formation.losses.hard_negative_mining import hard_negative_mining
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
import xarray as xr

# + [markdown] papermill={"duration": 0.003225, "end_time": "2022-09-13T14:34:43.343766", "exception": false, "start_time": "2022-09-13T14:34:43.340541", "status": "completed"} tags=[]
# # Predict TC Formation using Grid Probability

# + [markdown] papermill={"duration": 0.003039, "end_time": "2022-09-13T14:34:43.349966", "exception": false, "start_time": "2022-09-13T14:34:43.346927", "status": "completed"} tags=[]
# Configurations to run for this experiment.

# + papermill={"duration": 1.502862, "end_time": "2022-09-13T14:34:44.855977", "exception": false, "start_time": "2022-09-13T14:34:43.353115", "status": "completed"} tags=[]
exp_name = 'tc_grid_prob_unet_baseline_12h'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/theanh_WPAC_baseline/tc_12h.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
print(train_path, val_path)
# test_path = data_path.replace('.csv', '_test.csv')
# subset = dict(
#     absvprs=[900, 750],
#     rhprs=[750],
#     tmpprs=[900, 500],
#     hgtprs=[500],
#     vvelprs=[500],
#     ugrdprs=[800, 200],
#     vgrdprs=[800, 200],
# )
subset = dict(
    hgtprs=[700, 500, 250],
    ugrdprs=[700, 500, 250],
    vgrdprs=[700, 500, 250],
    capesfc=None,
    absvprs=None,
    rhprs=None,
    tmpprs=None,
    vvelprs=None,
    tmpsfc=None,
    slp=None,
)
data_shape = (218, 434, 9)
use_softmax = False

# + [markdown] papermill={"duration": 0.003309, "end_time": "2022-09-13T14:34:44.863183", "exception": false, "start_time": "2022-09-13T14:34:44.859874", "status": "completed"} tags=[]
# Create U-Net model with normalization layer.

# + papermill={"duration": 5.161879, "end_time": "2022-09-13T14:34:50.028312", "exception": false, "start_time": "2022-09-13T14:34:44.866433", "status": "completed"} tags=[]
input_layer = keras.Input((218 // 2, 434 // 2, 9))
normalization_layer = preprocessing.Normalization()
model = unet.Unet(
    input_tensor=normalization_layer(input_layer),
    model_name='unet',
    classifier_activation='sigmoid' if not use_softmax else 'softmax',
    output_classes=1 if not use_softmax else 2,
    decoder_shortcut_mode='concat',
    filters_block=[64, 128, 256])
model.summary()

# + [markdown] papermill={"duration": 0.007043, "end_time": "2022-09-13T14:34:50.044308", "exception": false, "start_time": "2022-09-13T14:34:50.037265", "status": "completed"} tags=[]
# Then, we load the training and validation dataset.

# + papermill={"duration": 210.098208, "end_time": "2022-09-13T14:38:20.149448", "exception": false, "start_time": "2022-09-13T14:34:50.051240", "status": "completed"} tags=[]
tc_avg_radius_lat_deg = 3
data_loader = ts_data.TropicalCycloneWithGridProbabilityDataLoader(
    data_shape=data_shape,
    tc_avg_radius_lat_deg=tc_avg_radius_lat_deg,
    subset=subset,
    softmax_output=use_softmax,
    smooth_gt=True,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=128,
#     leadtimes=12,
    shuffle=True,
    # nonTCRatio=3,
)
validation = data_loader.load_dataset(val_path, batch_size=128)

# + [markdown] papermill={"duration": 0.007827, "end_time": "2022-09-13T14:38:20.165014", "exception": false, "start_time": "2022-09-13T14:38:20.157187", "status": "completed"} tags=[]
# After that, we will initialize the normalization layer,
# and compile the model.

# + papermill={"duration": 1375.639524, "end_time": "2022-09-13T15:01:15.812013", "exception": false, "start_time": "2022-09-13T14:38:20.172489", "status": "completed"} tags=[]
def remove_nans(x, y):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), y

def spatial_scale(x, y):
    # print(x.shape)
    return x[:, ::2, ::2], y[:, ::2, ::2]

training = training.map(spatial_scale).map(remove_nans)
validation = validation.map(spatial_scale).map(remove_nans)

features = training.map(lambda feature, _: feature)
normalization_layer.adapt(features)


# + papermill={"duration": 23.977399, "end_time": "2022-09-13T15:01:39.802008", "exception": false, "start_time": "2022-09-13T15:01:15.824609", "status": "completed"} tags=[]
@hard_negative_mining
def hard_negative_mined_sigmoid_focal_loss(y_true, y_pred):
    fl = tfa.losses.SigmoidFocalCrossEntropy()
    return fl(y_true, y_pred)

@hard_negative_mining
def hard_negative_mined_binary_crossentropy_loss(y_true, y_pred):
    l = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return l(y_true, y_pred)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def combine_loss_funcs(*fns):
    def combined_loss(y_true, y_pred):
        return sum(f(y_true, y_pred) for f in fns)
    
    return combined_loss

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(),
    # loss=combine_loss_funcs(hard_negative_mined_sigmoid_focal_loss, dice_loss),
    loss=dice_loss,
    # loss=hard_negative_mined_sigmoid_focal_loss,
    # loss=hard_negative_mined_binary_crossentropy_loss,
    metrics=[
        'binary_accuracy',
        keras.metrics.Recall(name='recall', class_id=1 if use_softmax else None),
        keras.metrics.Precision(name='precision', class_id=1 if use_softmax else None),
        tfm.CustomF1Score(name='f1', class_id=1 if use_softmax else None),
        bb.BBoxesIoUMetric(name='IoU', iou_threshold=0.2),
        #tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        #tfm.PrecisionScore(from_logits=True),
        #tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ])

# + [markdown] papermill={"duration": 0.007677, "end_time": "2022-09-13T15:01:39.817938", "exception": false, "start_time": "2022-09-13T15:01:39.810261", "status": "completed"} tags=[]
# Finally, we can train the model!

# + papermill={"duration": 10194.522386, "end_time": "2022-09-13T17:51:34.347832", "exception": false, "start_time": "2022-09-13T15:01:39.825446", "status": "completed"} tags=[]
epochs = 500
model.fit(
    training,
    epochs=epochs,
    validation_data=validation,
    validation_freq=10,
    shuffle=True,
    callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_ckp_best_val",
            monitor='IoU',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor='IoU',
            mode='max',
            verbose=1,
            patience=50,
            restore_best_weights=True
        ),
    ]
)

# + papermill={"duration": 19.142783, "end_time": "2022-09-13T17:51:53.807687", "exception": false, "start_time": "2022-09-13T17:51:34.664904", "status": "completed"} tags=[]
# testing = data_loader.load_dataset(
#     test_path,
#     batch_size=64,
# ).map(spatial_scale).map(remove_nans)
# model.evaluate(testing)
