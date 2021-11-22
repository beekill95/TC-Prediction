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
sys.path.append('../..')  # noqa

from tc_formation import data, plot
import tc_formation.models.layers
import tc_formation.models.unet as unet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
# -

# # UNet

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

exp_name = 'baseline_resnet_multileadtime'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb'
#data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_700mb'
# data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb'
data_path = '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
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

model = unet.Unet(
    input_shape=data_shape,
    classifier_activation=None,)
model.summary()

