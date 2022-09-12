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

# + papermill={"duration": 0.107099, "end_time": "2022-07-25T14:45:27.843198", "exception": false, "start_time": "2022-07-25T14:45:27.736099", "status": "completed"} tags=[]
# %cd ../..
# %load_ext autoreload
# %autoreload 2

# + papermill={"duration": 51.33544, "end_time": "2022-07-25T14:46:19.183646", "exception": false, "start_time": "2022-07-25T14:45:27.848206", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.004526, "end_time": "2022-07-25T14:46:19.194441", "exception": false, "start_time": "2022-07-25T14:46:19.189915", "status": "completed"} tags=[]
# # Predict TC Formation using Grid Probability

# + [markdown] papermill={"duration": 0.00445, "end_time": "2022-07-25T14:46:19.203327", "exception": false, "start_time": "2022-07-25T14:46:19.198877", "status": "completed"} tags=[]
# Configurations to run for this experiment.

# + papermill={"duration": 2.641352, "end_time": "2022-07-25T14:46:21.849370", "exception": false, "start_time": "2022-07-25T14:46:19.208018", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.004656, "end_time": "2022-07-25T14:46:21.862357", "exception": false, "start_time": "2022-07-25T14:46:21.857701", "status": "completed"} tags=[]
# Create U-Net model with normalization layer.

# + papermill={"duration": 6.377111, "end_time": "2022-07-25T14:46:28.244075", "exception": false, "start_time": "2022-07-25T14:46:21.866964", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.00509, "end_time": "2022-07-25T14:46:28.255254", "exception": false, "start_time": "2022-07-25T14:46:28.250164", "status": "completed"} tags=[]
# Then, we load the training and validation dataset.

# + papermill={"duration": 45.081445, "end_time": "2022-07-25T14:47:13.341782", "exception": false, "start_time": "2022-07-25T14:46:28.260337", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.005594, "end_time": "2022-07-25T14:47:13.353509", "exception": false, "start_time": "2022-07-25T14:47:13.347915", "status": "completed"} tags=[]
# After that, we will initialize the normalization layer,
# and compile the model.

# + papermill={"duration": 933.745589, "end_time": "2022-07-25T15:02:47.104730", "exception": false, "start_time": "2022-07-25T14:47:13.359141", "status": "completed"} tags=[]
def remove_nans(x, y):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), y

def spatial_scale(x, y):
    # print(x.shape)
    return x[:, ::2, ::2], y[:, ::2, ::2]

training = training.map(spatial_scale).map(remove_nans)
validation = validation.map(spatial_scale).map(remove_nans)

features = training.map(lambda feature, _: feature)
normalization_layer.adapt(features)


# + papermill={"duration": 7.338589, "end_time": "2022-07-25T15:02:54.451538", "exception": false, "start_time": "2022-07-25T15:02:47.112949", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.006268, "end_time": "2022-07-25T15:02:54.464786", "exception": false, "start_time": "2022-07-25T15:02:54.458518", "status": "completed"} tags=[]
# Finally, we can train the model!

# + papermill={"duration": 5968.20456, "end_time": "2022-07-25T16:42:22.675455", "exception": false, "start_time": "2022-07-25T15:02:54.470895", "status": "completed"} tags=[]
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
            monitor='val_IoU',
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

# + papermill={"duration": 1392.589766, "end_time": "2022-07-25T17:05:35.299304", "exception": false, "start_time": "2022-07-25T16:42:22.709538", "status": "completed"} tags=[]
# testing = data_loader.load_dataset(
#     test_path,
#     batch_size=64,
# ).map(spatial_scale).map(remove_nans)
# model.evaluate(testing)
