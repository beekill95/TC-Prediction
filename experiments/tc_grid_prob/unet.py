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

# %cd ../..

from datetime import datetime
from tc_formation.models import unet
from tc_formation import tf_metrics as tfm
from tc_formation.data import data
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa

# # Predict TC Formation using Grid Probability

# Configurations to run for this experiment.

exp_name = 'tc_grid_prob_unet'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h.csv'
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

# Create U-Net model with normalization layer.

input_layer = keras.Input(data_shape)
normalization_layer = preprocessing.Normalization()
model = unet.Unet(input_tensor=normalization_layer(input_layer), model_name='unet')
model.summary()

# Then, we load the training and validation dataset.

# +
import tensorflow as tf

training = data.load_data_with_tc_probability(
    train_path,
    data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset)
validation = data.load_data_with_tc_probability(val_path, data_shape, subset=subset)

it = iter(training)
for i in range(2):
    _, prob = next(it)
    print(prob)
# -

# After that, we will initialize the normalization layer,
# and compile the model.

# +
features = training.map(lambda feature, _: feature)
normalization_layer.adapt(features)

model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True),
    metrics=[
        'binary_accuracy',
        #tfm.RecallScore(from_logits=True),
        #tfm.PrecisionScore(from_logits=True),
        #tfm.F1Score(num_classes=1, from_logits=True, threshold=0.5),
    ])
# -

# Finally, we can train the model!

epochs = 150
model.fit(
    training,
    epochs=epochs,
    validation_data=validation,
    shuffle=True)
