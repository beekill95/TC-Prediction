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

from tc_formation import plot
from tc_formation.data import formation_prediction as data
import tc_formation.models.layers
import tc_formation.models.resnet as resnet
import tc_formation.tf_metrics as tfm
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from datetime import datetime

# # Use ResNet to predict Focused TC Masked

# The data that we're using will have the following shape.
# Should change it to whatever the shape of the data we're going to use down there.

# ## Experiment Specifications

# +
exp_name = 'resnet_focused_tc_location'
runtime = datetime.now().strftime('%Y_%b_%d_%H_%M')
data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
train_path = data_path.replace('.csv', '_train.csv')
val_path = data_path.replace('.csv', '_val.csv')
test_path = data_path.replace('.csv', '_test.csv')

# Original features subset.
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

subset = dict(
    absvprs=[1000, 925, 850, 500, 300, 250],
    rhprs=[1000, 925, 850, 500, 300, 250],
    tmpprs=[1000, 925, 850, 500, 300, 250],
    hgtprs=[1000, 925, 850, 500, 300, 250],
    vvelprs=[1000, 925, 850, 500, 300, 250],
    ugrdprs=[1000, 925, 850, 500, 300, 250],
    vgrdprs=[1000, 925, 850, 500, 300, 250],
)
data_shape = (41, 161, 44)
# -

# ## Data

# +
data_loader = data.FocusedTCFormationDataLoader(
    data_shape=data_shape,
    subset=subset,
    tc_avg_radius_lat_deg=5,
)

full_training = data_loader.load_dataset(train_path)
validation = data_loader.load_dataset(val_path)

# +
# Create data normalization to transform data into 0 mean and 1 standard deviation.
normalizer = preprocessing.Normalization(axis=-1)

for X, _, _ in iter(full_training):
    normalizer.adapt(X)


# +
def normalize_data_and_focused_on_will_happen_tc(X, y, mask):
    return normalizer(X) * mask, y

full_training = full_training.map(normalize_data_and_focused_on_will_happen_tc)
validation = validation.map(normalize_data_and_focused_on_will_happen_tc)
# -

# ## Model

# + tags=[]
model = resnet.ResNet18v2(
    input_shape=data_shape,
    include_top=True,
    classes=1,
    classifier_activation=None,)
model.summary()
# -

# Build the model using BinaryCrossentropy loss

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

# Load our training and validation data.

# ### Training

# + tags=[]
epochs = 150
first_stage_history = model.fit(
    full_training,
    epochs=epochs,
    validation_data=validation,
    #class_weight={1: 3., 0: 1.},
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            # Setting patience so low because the model can achieve very good accuracy,
            # in very little time.
            # So setting 5 here to save us some waiting time.
            patience=5,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_val",
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{exp_name}_{runtime}_1st_ckp_best_train",
            monitor='f1_score',
            mode='max',
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'outputs/{exp_name}_{runtime}_1st_board',
        ),
    ]
)

plot.plot_training_history(first_stage_history, "First stage training")
# -

# ### Testing

testing = data_loader.load_dataset(test_path)
testing = testing.map(normalize_data_and_focused_on_will_happen_tc)
model.evaluate(testing)

# ## Model Explanation Using Integrated Gradients

# ### Generate Testing Predictions

# +
import numpy as np # noqa

# Using the model to predict the testing dataset.
pred = model.predict(testing)
pred = 1 / (1 + np.exp(-pred))

# +
import tc_formation.data.label as label # noqa

test_tc = label.load_label(test_path)
print(len(test_tc), len(pred))
test_tc['Pred Prob'] = pred
test_tc['Predicted'] = test_tc['Pred Prob'] >= 0.5

matched = test_tc[test_tc['TC'] == test_tc['Predicted']]
difference = test_tc[test_tc['TC'] != test_tc['Predicted']]

true_postivies = matched[matched['TC'] == 1]
true_postivies.head()
# -

# ### Baseline Input

baseline_input = np.zeros(data_shape)
pred = model.predict(normalizer(np.asarray([baseline_input])))
1 / (1 + np.exp(-pred))

# ### Integrated Gradients

# Augment model to return probability prediction,
# not just logits.
sigmoid = 1 / (1 + tf.exp(-model.outputs[0]))
aug_model = keras.Model(model.inputs, sigmoid)

# +
import matplotlib.pyplot as plt # noqa
from tc_formation.plots import observations as plt_obs # noqa
from tc_formation.plots import decorators as _d # noqa
from tc_formation.model_explanation import integrated_gradient as IG # noqa
from tc_formation.plots.integrated_gradient_visualizer import IntegratedGradientVisualizer # noqa

@_d._with_axes
@_d._with_basemap
def plot_SST(dataset, basemap=None, ax=None, **kwargs):
    lats, longs = np.meshgrid(dataset['lon'], dataset['lat'])
    cs = basemap.contour(lats, longs, dataset['tmpsfc'], levels=np.arange(270, 310, 2), cmap='Reds')
    ax.clabel(cs, inline=True, fontsize=20)

def plot_stuff(ds, pressure_level, ax):
    # Plot Relative Humidity
#     plt_obs.plot_variable(
#         dataset=ds,
#         variable='rhprs',
#         pressure_level=pressure_level,
#         cmap='Blues',
#         ax=ax,
#         contourf_kwargs=dict(levels=np.arange(0, 110, 5)))
    
    # Plot wind field.
    plt_obs.plot_wind(dataset=ds, pressure_level=pressure_level, ax=ax, skip=4)

    # Plot SST
    plot_SST(dataset=ds, ax=ax)


# +
from ast import literal_eval # noqa
from tc_formation.model_explanation import integrated_gradient as IG # noqa
from tc_formation.plots.integrated_gradient_visualizer import IntegratedGradientVisualizer # noqa
import xarray as xr # noqa

def plot_samples(true_positives_df):
    visualizer = IntegratedGradientVisualizer()

    for _, row in true_positives_df.iterrows():
        ds = xr.open_dataset(row['Path'])
        other_tc_locations = literal_eval(row['Other TC Locations'])
        X, _, mask = data_loader.load_single_data(row['Path'], row['TC'], row['Latitude'], row['Longitude'])
        print(row['Path'], row['Pred Prob'], row['Predicted'])
        print('First Observed: ', row['First Observed'])
        lon = row["Longitude"]
        print(f'Location: {row["Latitude"]} lat - {lon if lon < 180 else lon - 360} lon')
        print(f'Other TC Locations: {other_tc_locations}')

        # Load data, predict and calculate integrated gradient.
        igrads = IG.integrated_gradient(
            aug_model,
            X, baseline_input,
            preprocessor=lambda x: normalizer(x) * mask.astype(np.float32))
        X = normalizer(np.asarray([X]))
        X *= mask
        a = model.predict(X)
        print(1/(1 + np.exp(-a)))
        print('Model Prediction: ', aug_model.predict(X))

        # Plot stuffs.
        fig, ax = plt.subplots(nrows=2, figsize=(30, 18))
        ax[0].set_title('SST and RH at 850mb, and Model Spatial Attribution')

        # Plot integrated result.
        a = visualizer.visualize(
            dataset=ds,
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=95,
            clip_below_percentile=28,
            morphological_cleanup=True,
            outlines=False,
            ax=ax[0],
        )

        plot_stuff(ds, 850, ax=ax[0])
        
        # Plot our mask to make sure it is correct!
        cs = ax[1].imshow(mask)
        ax[1].invert_yaxis()
        fig.colorbar(cs, ax=ax[1], fraction=0.012, pad=0.015)

        # Display the resulting plot.
        fig.tight_layout()
        display(fig)
        plt.close(fig)


# -

# #### True Positives

plot_samples(true_postivies.sample(10))

# #### True Negatives

# True Negative Samples
true_negatives = matched[matched['TC'] == 0]
plot_samples(true_negatives.sample(10))
