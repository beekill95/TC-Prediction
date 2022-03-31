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

# +
from tc_formation.data import data
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# -

# # Feature Importance of ResNet 18 on Multiple Leadtime and Large Domain

# ## Data

# Specify location of the data, as well as data shape.

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
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
nb_features_to_select = 5

# Load data into memory.

full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
validation = data.load_data_v1(val_path, data_shape=data_shape, subset=subset)
testing = data.load_data_v1(test_path, data_shape=data_shape, subset=subset)

# Perform data normalization and standardization.

normalizer = preprocessing.Normalization()
normalizer.adapt(full_training.map(lambda X, _: X))
# absvprs [900, 750]
# capesfc
# hgtprs [500]
# rhprs [750]
# tmpprs [900, 500]
# tmpsfc
# ugrdprs [800, 200]
# vgrdprs [800, 200]
# vvelprs [500]

# +
def normalize_data(X, y):
    return normalizer(X), y

full_training = full_training.map(normalize_data)
validation = validation.map(normalize_data)
testing = testing.map(normalize_data)
# -

# ## Model

def build_resnet_model(data_shape):
    model = resnet.ResNet18v2(
        input_shape=data_shape,
        include_top=True,
        classes=1,
        classifier_activation=None,)

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

    return model

# ## Sequential Feature Selection

# +
from tc_formation.features_selection.forward_features_selection import ForwardFeaturesSelection

initial_features = np.asarray([0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0.])

selector = ForwardFeaturesSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector.fit(full_training, validation, initial_features=initial_features)

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())

# +
selector1 = ForwardFeaturesSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector1.fit(full_training, validation, initial_features=initial_features)

print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())

# +
selector2 = ForwardFeaturesSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector2.fit(full_training, validation, initial_features=initial_features)

print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())
# -

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())
print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())
print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())
# absvprs [900, 750]
# capesfc
# hgtprs [500]
# rhprs [750]
# tmpprs [900, 500]
# tmpsfc
# ugrdprs [800, 200]
# vgrdprs [800, 200]
# vvelprs [500]

# $\Rightarrow$ It seems that the best 5 features are: 4 best features and tmpsfc
