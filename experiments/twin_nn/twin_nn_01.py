# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: tc_updated_2
#     language: python
#     name: tc_updated_2
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

from datetime import datetime
import numpy as np
from tc_formation.models import twin_nn
from tc_formation.data import data
import tensorflow as tf

# # Twin Neural Network

# ## Experiment Specification

exp_name = 'twin_nn_vortex_tc_removal_12h'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h_tc_removed/tc_ibtracs_12h_WP_EP_v4.csv'
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

# ## Data Loading

full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
    leadtime=[12],
    group_same_observations=False,
)
validation = data.load_data_v1(
    val_path,
    data_shape=data_shape,
    subset=subset,
    leadtime=[12],
    group_same_observations=True,
)

# Now, we will transform the datasets into
# datasets with two outputs.

# +
def convert_0_label_to_neg_1(X, y):
    y = tf.where(y == 0, -1, 1)
    return X, y


def duplicate_output(X, y):
    return X, dict(pos=y, neg=y)


full_training = (full_training
                 .map(convert_0_label_to_neg_1)
                 .map(duplicate_output))
validation = (validation
              .map(convert_0_label_to_neg_1)
              .map(duplicate_output))
# -

# ## Twin NN

model = twin_nn.twin_nn.TwinNN(
    input_shape=data_shape,
    fully_connected_hidden_layers=[100, 100],
    name='twin_nn',
)
model.summary()

# ### Loss function

model.compile(
    optimizer='adam',
    loss=dict(
        pos=twin_nn.loss.TwinNNLoss(label=1, C=2.0),
        neg=twin_nn.loss.TwinNNLoss(label=-1, C=2.0),
    ),
)

# ### Model Training

for X, y in iter(full_training):
    print(f'{X.shape=}')
    for name, value in y.items():
        print(f'y,{name=},{value.shape=}')
    break

epochs = 150
model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    shuffle=True,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
        )
    ],
)

# ## Testing

# ### Verification on Training Data

full_training = data.load_data_v1(
    train_path,
    data_shape=data_shape,
    batch_size=64,
    shuffle=True,
    subset=subset,
    leadtime=[12],
    group_same_observations=False,
)
full_training = full_training.map(convert_0_label_to_neg_1)
model.evaluate(full_training)

validation = data.load_data_v1(
    val_path,
    data_shape=data_shape,
    subset=subset,
    leadtime=[12],
    group_same_observations=True,
)
validation = validation.map(convert_0_label_to_neg_1)
model.evaluate(validation)

# ### Verification on Testing Data

testing = data.load_data_v1(
    test_path,
    data_shape=data_shape,
    subset=subset,
    group_same_observations=True,
    leadtime=[12],
)
testing = testing.map(convert_0_label_to_neg_1)
model.evaluate(testing)

# +
raw_pred = model.predict_raw(testing)
y_true = np.concatenate([y for _, y in testing.as_numpy_iterator()]).flatten()
print(y_true.shape)
print('==== Positive Labels ====')
for y, pos_dist, neg_dist in zip(y_true, raw_pred['pos'], raw_pred['neg']):
    if y == 1:
        print(f'Label {y=} with distance to pos class {pos_dist} and neg class {neg_dist}')

print('==== Negative Labels ====')
for y, pos_dist, neg_dist in zip(y_true, raw_pred['pos'], raw_pred['neg']):
    if y == -1:
        print(f'Label {y=} with distance to pos class {pos_dist} and neg class {neg_dist}')
