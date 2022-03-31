# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ../..

from tc_formation import plot
from tc_formation.data import formation_prediction as data
import tc_formation.models.layers
import tc_formation.models.multiscale_baseline as baseline
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime

# # Multi-Scale Prediction Baseline

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

# ## Experiment Specifications

# +
exp_name = 'multiscale_baseline'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')

# Original features subset.
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

subset = dict(
    absvprs=[1000, 925, 850, 700, 500, 300, 250],
    rhprs=[1000, 925, 850, 700, 500, 300, 250],
    tmpprs=[1000, 925, 850, 700, 500, 300, 250],
    hgtprs=[1000, 925, 850, 700, 500, 300, 250],
    vvelprs=[1000, 925, 850, 700, 500, 300, 250],
    ugrdprs=[1000, 925, 850, 700, 500, 300, 250],
    vgrdprs=[1000, 925, 850, 700, 500, 300, 250],
)
data_shape = (41, 161, 51)
# -

# ## Data

# +
data_loader = data.FocusedTCFormationDataLoader(
    data_shape=data_shape,
    subset=subset,
    tc_avg_radius_lat_deg=3,
)

full_training = data_loader.load_dataset(
    train_path,
    other_happening_tc_ratio=0)
validation = data_loader.load_dataset(val_path, other_happening_tc_ratio=0)

# +
# FIXME: currently, I don't want to normalize it.
# Create data normalization to transform data into 0 mean and 1 standard deviation.
# normalizer = preprocessing.Normalization(axis=-1)
# 
# for X, _, _ in iter(full_training):
#     normalizer.adapt(X)
# 
# def normalize_data_and_focused_on_will_happen_tc(X, y, mask):
#     return normalizer(X) * mask, y
# 
# full_training = full_training.map(normalize_data_and_focused_on_will_happen_tc)
# validation = validation.map(normalize_data_and_focused_on_will_happen_tc)
# -

# Now, we will create a set of expected outputs,
# because the Multi-Scale model has 4 output in total.

# +
import numpy as np # noqa

def create_set_of_expected_outputs(X, y, mask):
    half_shape = (np.asarray(X.shape[1:-1]) / 2).astype(dtype=np.int32)
    output_1 = tf.image.resize(mask, size=half_shape, method='nearest')

    half_shape = np.asarray(half_shape / 2, dtype=np.int32)
    output_2 = tf.image.resize(output_1, half_shape, method='nearest')

    half_shape = np.asarray(half_shape / 2, dtype=np.int32)
    output_3 = tf.image.resize(output_2, half_shape, method='nearest')

    return (X,
            dict(output=y,
                 output_1=output_1,
                 output_2=output_2,
                 output_3=output_3,))

# Then, apply it to the full training and validation data.
full_training = full_training.map(create_set_of_expected_outputs)
validation = validation.map(create_set_of_expected_outputs)
# -

# Just to make sure that our expected outputs are correct,
# can be commented afterwards.

# +
import matplotlib.pyplot as plt # noqa

fig, axes = plt.subplots(nrows=3, figsize=(10, 12))

for _, outputs in full_training:
    plotted = False

    for output, output_1, output_2, output_3 in zip(*outputs.values()):
        if output == 1:
            plotted = True

            axes[0].imshow(output_1)
            axes[0].set_title(output_1.shape)
            axes[1].imshow(output_2)
            axes[1].set_title(output_2.shape)
            axes[2].imshow(output_3)
            axes[2].set_title(output_3.shape)
            break

    if plotted:
        break

fig.tight_layout()
display(fig)
plt.close(fig)
# -

# ## Model

# + tags=[]
model = baseline.MultiscaleBaseline(
    input_shape=data_shape,
    classes=1,
    output_activation='sigmoid',
    name='multiscale_baseline',
)
model.summary(line_length=120)


# -

# Build the model using BinaryCrossentropy loss

# +
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def for_fun_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    
    numerator = tf.abs(y_true  - y_pred)
    denominator = (y_true + 1e-10)*(1 - y_true + 1e-10)
    
    return tf.reduce_sum(numerator / denominator)

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=dict(
        output=for_fun_loss,
        # output=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # output=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        # output_1=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        # output_2=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        # output_3=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        # output_1=dice_loss,
        # output_2=dice_loss,
        # output_3=dice_loss,
        # output_3=for_fun_loss,
    ),
    loss_weights=dict(
        output=1.0,
        # output_1=1.0,
        # output_2=2.0,
        # output_3=1.0,
    ),
    metrics=dict(
        output=[
            tfm.RecallScore(from_logits=True, name='recall'),
            tfm.PrecisionScore(from_logits=True, name='precision'),
            tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5, name='f1'),
        ],
        # output_1=tfm.CustomF1Score(name='f1', from_logits=True),
        # output_2=tfm.CustomF1Score(name='f1', from_logits=True),
        output_3=tfm.CustomF1Score(name='f1'),
    ),
    # metrics=[
    #     'binary_accuracy',
    #     tfm.RecallScore(from_logits=True),
    #     tfm.PrecisionScore(from_logits=True),
    #     tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    # ]
)
# -

# Load our training and validation data.

# ### Training

# + tags=[]
epochs = 500
first_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    # class_weight={1: 3., 0: 1.},
    shuffle=True,
    callbacks=[
        # keras.callbacks.EarlyStopping(
        #     monitor='val_f1_score',
        #     mode='max',
        #     verbose=1,
        #     # Setting patience so low because the model can achieve very good accuracy,
        #     # in very little time.
        #     # So setting 5 here to save us some waiting time.
        #     patience=5,
        #     restore_best_weights=True),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_val",
        #     monitor='val_f1_score',
        #     mode='max',
        #     save_best_only=True,
        # ),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_train",
        #     monitor='f1_score',
        #     mode='max',
        #     save_best_only=True,
        # ),
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

# ### Testing

testing = data_loader.load_dataset(test_path)
testing = testing.map(create_set_of_expected_outputs)
model.evaluate(testing)
