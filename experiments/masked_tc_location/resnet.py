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
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime

# # Use ResNet with Masked TC Location

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

# ## Experiment Specifications

exp_name = 'resnet_masked_tc_location'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
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

# ## Data

# +
data_loader = data.TCFormationPredictionDataLoader(
    data_shape=data_shape,
    subset=subset,
    produce_other_tc_locations_mask=True,
)

full_training = data_loader.load_dataset(train_path)
validation = data_loader.load_dataset(val_path)

# +
# Create data normalization to transform data into 0 mean and 1 standard deviation.
normalizer = preprocessing.Normalization(axis=-1)

for X, _, _ in iter(full_training):
    normalizer.adapt(X)


# +
def normalize_data_and_remove_other_happening_tc(X, y, mask):
    return normalizer(X) * mask, y

full_training = full_training.map(normalize_data_and_remove_other_happening_tc)
validation = validation.map(normalize_data_and_remove_other_happening_tc)
# -

# ## Model

# + tags=[]
model = resnet.ResNet18v2(
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
        tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ]
)

# Load our training and validation data.

# ### Training

# + tags=[]
epochs = 150
first_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    #class_weight={1: 3., 0: 1.},
    shuffle=True,
    callbacks=[
        # keras.callbacks.EarlyStopping(
        #     monitor='val_f1_score',
        #     mode='max',
        #     verbose=1,
        #     patience=20,
        #     restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_val",
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_train",
            monitor='f1_score',
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

# ### Testing

testing = data_loader.load_dataset(test_path)
testing = testing.map(normalize_data_and_remove_other_happening_tc)
model.evaluate(testing)
