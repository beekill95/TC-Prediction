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

# + papermill={"duration": 0.153626, "end_time": "2022-02-02T02:32:27.190177", "exception": false, "start_time": "2022-02-02T02:32:27.036551", "status": "completed"} tags=[]
# %cd ../..

# + papermill={"duration": 62.89909, "end_time": "2022-02-02T02:33:30.114305", "exception": false, "start_time": "2022-02-02T02:32:27.215215", "status": "completed"} tags=[]
from tc_formation.data import data
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np

# + [markdown] papermill={"duration": 0.023765, "end_time": "2022-02-02T02:33:30.165275", "exception": false, "start_time": "2022-02-02T02:33:30.141510", "status": "completed"} tags=[]
# # Feature Importance of ResNet 18 on Multiple Leadtime and Large Domain

# + [markdown] papermill={"duration": 0.024128, "end_time": "2022-02-02T02:33:30.213441", "exception": false, "start_time": "2022-02-02T02:33:30.189313", "status": "completed"} tags=[]
# ## Data

# + [markdown] papermill={"duration": 0.02421, "end_time": "2022-02-02T02:33:30.261409", "exception": false, "start_time": "2022-02-02T02:33:30.237199", "status": "completed"} tags=[]
# Specify location of the data, as well as data shape.

# + papermill={"duration": 0.030489, "end_time": "2022-02-02T02:33:30.315999", "exception": false, "start_time": "2022-02-02T02:33:30.285510", "status": "completed"} tags=[]
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
nb_features_to_select = 4

# + [markdown] papermill={"duration": 0.02428, "end_time": "2022-02-02T02:33:30.365558", "exception": false, "start_time": "2022-02-02T02:33:30.341278", "status": "completed"} tags=[]
# Load data into memory.

# + papermill={"duration": 5.799017, "end_time": "2022-02-02T02:33:36.188174", "exception": false, "start_time": "2022-02-02T02:33:30.389157", "status": "completed"} tags=[]
full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
)
validation = data.load_data_v1(val_path, data_shape=data_shape, subset=subset)
testing = data.load_data_v1(test_path, data_shape=data_shape, subset=subset)

# + [markdown] papermill={"duration": 0.026157, "end_time": "2022-02-02T02:33:36.240950", "exception": false, "start_time": "2022-02-02T02:33:36.214793", "status": "completed"} tags=[]
# Perform data normalization and standardization.

# + papermill={"duration": 205.459029, "end_time": "2022-02-02T02:37:01.726545", "exception": false, "start_time": "2022-02-02T02:33:36.267516", "status": "completed"} tags=[]
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

# + papermill={"duration": 0.081069, "end_time": "2022-02-02T02:37:01.838094", "exception": false, "start_time": "2022-02-02T02:37:01.757025", "status": "completed"} tags=[]
def normalize_data(X, y):
    return normalizer(X), y

full_training = full_training.map(normalize_data)
validation = validation.map(normalize_data)
testing = testing.map(normalize_data)

# + [markdown] papermill={"duration": 0.027419, "end_time": "2022-02-02T02:37:01.894991", "exception": false, "start_time": "2022-02-02T02:37:01.867572", "status": "completed"} tags=[]
# ## Model

# + papermill={"duration": 0.032661, "end_time": "2022-02-02T02:37:01.954406", "exception": false, "start_time": "2022-02-02T02:37:01.921745", "status": "completed"} tags=[]
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

# + [markdown] papermill={"duration": 0.026929, "end_time": "2022-02-02T02:37:02.008464", "exception": false, "start_time": "2022-02-02T02:37:01.981535", "status": "completed"} tags=[]
# ## Sequential Feature Selection

# + papermill={"duration": 0.036462, "end_time": "2022-02-02T02:37:02.071446", "exception": false, "start_time": "2022-02-02T02:37:02.034984", "status": "completed"} tags=[]
class SequentialFeatureSelection:
    def __init__(self, model_fn, data_shape, nb_features_to_select):
        self._init_data_shape = data_shape
        self._nb_feature_channels = data_shape[-1]
        self._model_fn = model_fn
        self._nb_features_to_select = nb_features_to_select
        self._best_proposal = None
        self._best_proposal_score = 0.0

    def best_proposal(self):
        return self._best_proposal
    
    def best_proposal_score(self):
        return self._best_proposal_score

    def fit(self, training, validation):
        best_objective_value = 0.0
        best_proposal = np.zeros(self._nb_feature_channels)
        proposals = self._propose_feature_masks(best_proposal)
        
        while best_proposal.sum() < self._nb_features_to_select:
            print(f'Evaluating proposal with nb features: {best_proposal.sum() + 1}')
            proposal_updated = False

            for proposal in proposals:
                print('\nCurrent proposal:', proposal)
                masked_training = training.map(lambda X, y: (X[:, :, :] * proposal, y))
                masked_validation = validation.map(lambda X, y: (X[:, :, :] * proposal, y))

                model = self._model_fn(self._init_data_shape)
                model.fit(
                    masked_training,
                    epochs=50,
                    validation_data=masked_validation,
                    class_weight={1: 10., 0: 1.},
                    shuffle=True,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_f1_score',
                            mode='max',
                            verbose=1,
                            patience=20,
                            restore_best_weights=True),
                    ],
                    verbose=0,
                )

                objective_value = model.evaluate(masked_validation)[4]
                if objective_value > best_objective_value:
                    print(f'Best proposal improve from {best_objective_value} to {objective_value}')
                    print(f'with proposal {proposal}')
                    best_objective_value = objective_value
                    best_proposal = proposal
                    proposal_updated = True


            if not proposal_updated:
                print('Proposal not updated. Stop!!')
                break
            print('== Propose next batch of masks from current best proposal', best_proposal)
            proposals = self._propose_feature_masks(best_proposal)

        # Save the best proposal.
        self._best_proposal = best_proposal
        self._best_proposal_score = best_objective_value

    def _propose_feature_masks(self, feature_mask):
        mask = ~(feature_mask > 0)
        proposals = []
        for i, m in enumerate(mask):
            if m:
                proposal = np.zeros(self._nb_feature_channels)
                proposal[i] = 1
                proposals.append(proposal + feature_mask)
                
        return proposals


# + papermill={"duration": 12053.537364, "end_time": "2022-02-02T05:57:55.635619", "exception": false, "start_time": "2022-02-02T02:37:02.098255", "status": "completed"} tags=[]
selector = SequentialFeatureSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector.fit(full_training, validation)

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())

# + papermill={"duration": 11465.937514, "end_time": "2022-02-02T09:09:02.288630", "exception": false, "start_time": "2022-02-02T05:57:56.351116", "status": "completed"} tags=[]
selector1 = SequentialFeatureSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector1.fit(full_training, validation)

print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())

# + papermill={"duration": 12156.656613, "end_time": "2022-02-02T12:31:40.323110", "exception": false, "start_time": "2022-02-02T09:09:03.666497", "status": "completed"} tags=[]
selector2 = SequentialFeatureSelection(
    model_fn=build_resnet_model,
    data_shape=data_shape,
    nb_features_to_select=nb_features_to_select)
selector2.fit(full_training, validation)

print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())

# + papermill={"duration": 2.072577, "end_time": "2022-02-02T12:31:44.457535", "exception": false, "start_time": "2022-02-02T12:31:42.384958", "status": "completed"} tags=[]
print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())
print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())
print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())
# -

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
# $\Rightarrow$ The best features are tmpsfc, ugrdprs @800 & @200, vgrdprs @800
