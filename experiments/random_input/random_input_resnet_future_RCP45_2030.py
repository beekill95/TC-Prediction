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
import tensorflow.keras as keras
from scipy.stats import bernoulli
from scipy.special import expit
import sklearn.metrics as skmetrics
# -

# # Random Input

# Load model.
model_path = 'outputs/baseline_resnet_theanh_RCP45_2030_2022_Aug_27_11_49_1st_ckp/'
model = keras.models.load_model(model_path, compile=False)
model.trainable = False
model.summary()

# Specify data.
data_shape = (114, 219, 12)
batch_size = 64
positive_negative_ratio = 0.25
num_batches = 64

# Apply model on randomly generated data.

random_true_target = []
pred_target = []
for batch in range(num_batches):
    data = np.random.normal(size=(batch_size,) + data_shape)
    target = bernoulli.rvs(positive_negative_ratio, size=(batch_size,))

    logits = model.predict(data)
    probs = expit(logits)
    pred = np.where(probs > 0.5, 1, 0)

    # Store results.
    random_true_target.extend(target)
    pred_target.extend(pred)

# Calculate the results.

# +
f1 = skmetrics.f1_score(random_true_target, pred_target)
precision = skmetrics.precision_score(random_true_target, pred_target)
recall = skmetrics.recall_score(random_true_target, pred_target)

print(f'{f1=}, {recall=}, {precision=}')
