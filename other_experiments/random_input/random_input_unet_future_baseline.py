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
#     display_name: tc_updated_2
#     language: python
#     name: tc_updated_2
# ---

# +
# %cd ../..

import numpy as np
import tc_formation
import tc_formation.utils.unet_track as unet_track
import tensorflow.keras as keras
from scipy.stats import bernoulli
from scipy.special import expit
import sklearn.metrics as skmetrics
# -

# # Random Input

# Load model.
model_path = 'outputs/tc_grid_prob_unet_baseline_12h_2022_Sep_13_10_34_ckp_best_val/'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

# Specify data.
data_shape = (109, 217, 9)
batch_size = 64
positive_negative_ratio = 0.25
num_batches = 10

# Apply model on randomly generated data.

# +
import matplotlib.pyplot as plt

pred_centers = []
center_locator = unet_track.UnetPredictionCenter()
for batch in range(num_batches):
    data = np.random.normal(size=(batch_size,) + data_shape)
    print(data.shape)

    probs = model.predict(data)

    if len(pred_centers) <= 0:
        plt.figure()
        plt.imshow(probs[0])
        plt.figure()
        plt.imshow(probs[32])
        plt.figure()
        plt.imshow(probs[63])


    # for prob in probs:
    #     centers = center_locator.get_centers(probs)

    # Store results.
    pred_centers.extend(center_locator.get_centers(prob) for prob in probs)
# -

# Calculate the results.

# +
# f1 = skmetrics.f1_score(random_true_target, pred_target)
# precision = skmetrics.precision_score(random_true_target, pred_target)
# recall = skmetrics.recall_score(random_true_target, pred_target)

# print(f'{f1=}, {recall=}, {precision=}')
print(pred_centers)
