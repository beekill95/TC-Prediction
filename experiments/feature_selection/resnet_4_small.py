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

# + papermill={"duration": 0.145977, "end_time": "2022-02-10T02:30:06.052078", "exception": false, "start_time": "2022-02-10T02:30:05.906101", "status": "completed"} tags=[]
# %cd ../..

# + papermill={"duration": 69.174683, "end_time": "2022-02-10T02:31:15.252129", "exception": false, "start_time": "2022-02-10T02:30:06.077446", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.025598, "end_time": "2022-02-10T02:31:15.305105", "exception": false, "start_time": "2022-02-10T02:31:15.279507", "status": "completed"} tags=[]
# # Feature Importance of ResNet 18 on Multiple Leadtime and Small Domain

# + [markdown] papermill={"duration": 0.025571, "end_time": "2022-02-10T02:31:15.356675", "exception": false, "start_time": "2022-02-10T02:31:15.331104", "status": "completed"} tags=[]
# ## Data

# + [markdown] papermill={"duration": 0.026785, "end_time": "2022-02-10T02:31:15.409443", "exception": false, "start_time": "2022-02-10T02:31:15.382658", "status": "completed"} tags=[]
# Specify location of the data, as well as data shape.

# + papermill={"duration": 0.031815, "end_time": "2022-02-10T02:31:15.467199", "exception": false, "start_time": "2022-02-10T02:31:15.435384", "status": "completed"} tags=[]
data_path = 'data/nolabels_wp_only_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
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
data_shape = (41, 81, 13)

# + [markdown] papermill={"duration": 0.025743, "end_time": "2022-02-10T02:31:15.518722", "exception": false, "start_time": "2022-02-10T02:31:15.492979", "status": "completed"} tags=[]
# Load data into memory.

# + papermill={"duration": 4.833546, "end_time": "2022-02-10T02:31:20.377668", "exception": false, "start_time": "2022-02-10T02:31:15.544122", "status": "completed"} tags=[]
full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
validation = data.load_data_v1(val_path, data_shape=data_shape, subset=subset)
testing = data.load_data_v1(test_path, data_shape=data_shape, subset=subset)

# + [markdown] papermill={"duration": 0.027412, "end_time": "2022-02-10T02:31:20.433457", "exception": false, "start_time": "2022-02-10T02:31:20.406045", "status": "completed"} tags=[]
# Perform data normalization and standardization.

# + papermill={"duration": 226.945355, "end_time": "2022-02-10T02:35:07.406248", "exception": false, "start_time": "2022-02-10T02:31:20.460893", "status": "completed"} tags=[]
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

# + papermill={"duration": 0.100042, "end_time": "2022-02-10T02:35:07.535867", "exception": false, "start_time": "2022-02-10T02:35:07.435825", "status": "completed"} tags=[]
def normalize_data(X, y):
    return normalizer(X), y

full_training = full_training.map(normalize_data)
validation = validation.map(normalize_data)
testing = testing.map(normalize_data)

# + [markdown] papermill={"duration": 0.028323, "end_time": "2022-02-10T02:35:07.593175", "exception": false, "start_time": "2022-02-10T02:35:07.564852", "status": "completed"} tags=[]
# ## Model

# + papermill={"duration": 0.033709, "end_time": "2022-02-10T02:35:07.656038", "exception": false, "start_time": "2022-02-10T02:35:07.622329", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.028312, "end_time": "2022-02-10T02:35:07.712514", "exception": false, "start_time": "2022-02-10T02:35:07.684202", "status": "completed"} tags=[]
# ## Sequential Feature Selection

# + papermill={"duration": 3042.467217, "end_time": "2022-02-10T03:25:50.274487", "exception": false, "start_time": "2022-02-10T02:35:07.807270", "status": "completed"} tags=[]
from tc_formation.features_selection.forward_features_selection import ForwardFeaturesSelection

initial_features = np.asarray([0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.])

selector = ForwardFeaturesSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector.fit(full_training, validation, initial_features=initial_features)

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())

# + papermill={"duration": 2922.161567, "end_time": "2022-02-10T04:14:32.756125", "exception": false, "start_time": "2022-02-10T03:25:50.594558", "status": "completed"} tags=[]
selector1 = SequentialFeatureSelection(model_fn=build_resnet_model, data_shape=data_shape, nb_features_to_select=3)
selector1.fit(full_training, validation)

print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())

# + papermill={"duration": 2921.470884, "end_time": "2022-02-10T05:03:14.873708", "exception": false, "start_time": "2022-02-10T04:14:33.402824", "status": "completed"} tags=[]
selector2 = SequentialFeatureSelection(model_fn=build_resnet_model, data_shape=data_shape, nb_features_to_select=3)
selector2.fit(full_training, validation)

print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())

# + papermill={"duration": 1.009526, "end_time": "2022-02-10T05:03:16.885046", "exception": false, "start_time": "2022-02-10T05:03:15.875520", "status": "completed"} tags=[]
print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())
print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())
print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())

# + [markdown] papermill={"duration": 0.994031, "end_time": "2022-02-10T05:03:18.874986", "exception": false, "start_time": "2022-02-10T05:03:17.880955", "status": "completed"} tags=[]
# * absvprs [900, 750]
# * capesfc
# * hgtprs [500]
# * rhprs [750]
# * tmpprs [900, 500]
# * tmpsfc
# * ugrdprs [800, 200]
# * vgrdprs [800, 200]
# * vvelprs [500]
#
# It seems that the best three features are: capesfc, ugrdprs @ 800 and vgrdprs @ 800
# -

# $\Rightarrow$ The features are Capesfc, velprs, and ugrdprs @ 200
