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

from collections import OrderedDict
from tc_formation.data import data
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime
import numpy as np

# # Feature Importance of ResNet 18 on Multiple Leadtime and Large Domain

# ## Data

# Specify location of the data, as well as data shape.

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_6h_12h_18h_24h_30h_36h_42h_48h.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')
# subset = dict(
#     absvprs=[900, 750],
#     rhprs=[750],
#     tmpprs=[900, 500],
#     hgtprs=[500],
#     vvelprs=[500],
#     ugrdprs=[800, 200],
#     vgrdprs=[800, 200],
# )
subset = OrderedDict(
    absvprs=[900, 750],
    capesfc=True,
    hgtprs=[500],
    rhprs=[750],
    tmpprs=[900, 500],
    tmpsfc=True,
    ugrdprs=[800, 200],
    vgrdprs=[800, 200],
    vvelprs=[500],
)
data_shape = (41, 161, 13)

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


# +
selector = SequentialFeatureSelection(model_fn=build_resnet_model, data_shape=data_shape, nb_features_to_select=3)
selector.fit(full_training, validation)

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())

# +
selector1 = SequentialFeatureSelection(model_fn=build_resnet_model, data_shape=data_shape, nb_features_to_select=3)
selector1.fit(full_training, validation)

print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())

# +
selector2 = SequentialFeatureSelection(model_fn=build_resnet_model, data_shape=data_shape, nb_features_to_select=3)
selector2.fit(full_training, validation)

print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())
# -

print('Best proposal: ', selector.best_proposal(), ' with score: ', selector.best_proposal_score())
print('Best proposal: ', selector1.best_proposal(), ' with score: ', selector1.best_proposal_score())
print('Best proposal: ', selector2.best_proposal(), ' with score: ', selector2.best_proposal_score())

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
