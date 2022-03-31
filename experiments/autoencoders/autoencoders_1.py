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

# + tags=[]
from datetime import timedelta # noqa
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
        time_delta=timedelta(hours=12),
        subset=subset)
# -

# ## AutoEncoder Model

# +
from tc_formation.autoencoders import autoencoders # noqa
import tensorflow.keras as keras # noqa
from tensorflow.keras.layers.experimental import preprocessing # noqa

inputs = keras.layers.Input(data_shape)
normalizer = preprocessing.Normalization()

x = normalizer(inputs)

model = autoencoders.AutoEncoders(
        input_shape=data_shape,
        input_tensor=x,
        name='autoencoders')
x = model.outputs[0]
x = normalizer.variance * x + normalizer.mean

model = keras.Model(inputs, x)
model.summary()
# -

# ## Training

# ### Prepare the dataset

# +
def crop_Y(dataset):
    return dataset.map(lambda X, Y: (X, Y[:-1, :-1]))
    
def prepare_dataset(dataset, batch_size):
    return (dataset
            .batch(batch_size)
            .prefetch(1))


(training,
 validation,
 testing) = tuple(
         map(lambda ds: prepare_dataset(ds, 64), (training, validation, testing)))
# -

# ### Fit Normalizer Layer

for X, _ in training:
    normalizer.adapt(X)

# ### Compile model and Training

model.compile(
    optimizer='adam',
    loss='mse',
)

# + tags=[]
model.fit(
    training,
    epochs=1000,
    validation_data=validation,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=20),
    ],
)
# -

# ## Testing

model.evaluate(testing)

# +
import matplotlib.pyplot as plt # noqa
import numpy as np # noqa

for X, Y in iter(testing):
    X = X[0]
    Y = Y[0]
    
    Y_pred, = model.predict(np.asarray([X]))
    
    for i in range(Y.shape[-1]):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(30, 6))
        im = ax1.imshow(Y[:, :, i])
        fig.colorbar(im, ax=ax1, fraction=0.02)
        ax1.set_title('Truth')
        
        im = ax2.imshow(Y_pred[:, :, i])
        fig.colorbar(im, ax=ax2, fraction=0.02)
        fig.tight_layout()
        ax2.set_title('Prediction')
        
        im = ax3.imshow(Y[:, :, i] - Y_pred[:, :, i])
        fig.colorbar(im, ax=ax3, fraction=0.02)
        fig.tight_layout()
        ax3.set_title('Error')
        
        display(fig)
        plt.close(fig)
    
    break
