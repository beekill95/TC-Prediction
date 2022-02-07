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

# %cd ../../
# %load_ext autoreload
# %autoreload 2

# # Auto-Encoders

# ## Data

# +
from tc_formation.autoencoders import data as autoencoder_data # noqa

OBSERVATIONS_DIR = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h'

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
(training,
 validation,
 testing) = autoencoder_data.load_reconstruction_datasets(
        OBSERVATIONS_DIR,
        data_shape,
        subset=subset)
# -

# ## AutoEncoder Model

# +
from tc_formation.autoencoders import autoencoders # noqa
import tensorflow.keras as keras # noqa

model = autoencoders.AutoEncoders(
        input_shape=data_shape,
        name='autoencoders')
x = keras.layers.Cropping2D(((1, 2), (1, 2)))(model.outputs[0])
model = keras.Model(model.inputs, x)
model.summary()
# -

# ## Training

# +
def prepare_dataset(dataset, batch_size):
    return (dataset
            .batch(batch_size)
            .prefetch(1))

(training,
 validation,
 testing) = tuple(
         map(lambda ds: prepare_dataset(ds, 64), (training, validation, testing)))
# -

for X, Y in iter(training):
#     print(X[0, :5, :5])
    break

# +
model.compile(
    optimizer='adam',
    loss='mse',
)

model.fit(
    training,
    epochs=100,
    validation_data=validation,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20),
    ],
)
