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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %cd ../..
# %load_ext autoreload
# %autoreload 2

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tc_formation.binary_classifications.data.patches_with_genesis_tfrecords_data_loader import PatchesWithGenesisTFRecordDataLoader
from tc_formation.model_explanation import integrated_gradient as IG # noqa
from tc_formation.plots.integrated_gradient_visualizer import IntegratedGradientVisualizer # noqa
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import xarray as xr
# -

# # Integrated Gradients of 04_exp14
# ## Data

# +
val_path = 'data/ncep_WP_EP_0h_all_binary_patches_all_variables_developed_storms_removed_v2_Val.tfrecords'

dataloader = PatchesWithGenesisTFRecordDataLoader()
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256, for_analyzing=True)

input_shape = (31, 31, 136)
def set_shape(shape, batch=True):
    def _set_shape(X, y, *args):
        if batch:
            X.set_shape((None, ) + shape)
            y.set_shape((None, 1))
        else:
            X.set_shape(shape)
            y.set_shape((1,))

        return X, y, *args

    return _set_shape


val_patches_ds = val_patches_ds.map(set_shape(input_shape))


# -

# ## Model
#
# Load pretrained model.

# +
class F1(tf.keras.metrics.Metric):
    def __init__(self, thresholds, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self._precision = tf.keras.metrics.Precision(thresholds=thresholds)
        self._recall = tf.keras.metrics.Recall(thresholds=thresholds)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self._precision.update_state(y_true, y_pred, sample_weight)
        self._recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        return 2 * p * r / (p + r + 1e-6)
    
    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()


path = 'saved_models/binary_classification_all_patches/04_exp12_20230217_13_17_0.649'
model = tf.keras.models.load_model(path, compile=False)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        'binary_accuracy',
        tf.keras.metrics.Precision(thresholds=0.5),
        tf.keras.metrics.Recall(thresholds=0.5),
        F1(thresholds=0.5),
    ],
)
model.summary()
# -

# ### Model Predictions

# +
validation_predictions = []
for X, y, filenames, patch_locations in tqdm(iter(val_patches_ds), desc='Predicting on Validation'):
    y_pred = model.predict(X, verbose=False)
    validation_predictions.extend(
        [dict(
            filename=filename.numpy().decode('utf-8'),
            loc=loc.numpy(),
            X=Xi.numpy(),
            y=yi.numpy()[0],
            pred=yi_pred[0])
         for filename, loc, Xi, yi, yi_pred in zip(filenames, patch_locations, X, y, y_pred)])

validation_predictions = pd.DataFrame(validation_predictions)
validation_predictions['pred_label'] = validation_predictions['pred'].apply(lambda p: 1 if p > 0.5 else 0)
validation_predictions.head()


# +
def load_ibtracs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=(1,), na_filter=False)
    # Parse data column.
    df['Date_storm'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')

    # Convert LON.
    df['Lon_storm'] = df['LON'].apply(lambda l: l if l > 0 else 360 + l)
    df['Lat_storm'] = df['LAT']

    # We only care about some columns.
    df = df[['SID', 'Date_storm', 'Lat_storm', 'Lon_storm', 'BASIN']]
    return df


def parse_date_from_nc_filename(filename: str):
    FMT = '%Y%m%d_%H_%M'
    filename, _ = os.path.splitext(os.path.basename(filename))
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, FMT)


def find_nearest_storm_at_the_same_time(ibtracs_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    def _find_nearest_storm(storms_df: pd.DataFrame, patch_center: np.ndarray) -> tuple[pd.Series, float]:
        nearest_distance = np.inf
        nearest_storm = None

        for _, storm in storms_df.iterrows():
            storm_center = np.asarray([storm['Lat_storm'], storm['Lon_storm']])
            distance = np.linalg.norm(patch_center - storm_center)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_storm = storm

        return nearest_storm, nearest_distance

    prediction_df['Date_file'] = prediction_df['filename'].apply(parse_date_from_nc_filename)
    
    # It's better to shrink our domain,
    # since we know that we only detect storms in WP and EP.
    ibtracs_df = ibtracs_df[ibtracs_df['BASIN'].isin(['WP', 'EP'])]

    # Loop through each prediction to find the nearest storm on the same day, if any.
    results = []
    for _, pred_row in prediction_df.iterrows():
        storms_df = ibtracs_df[ibtracs_df['Date_storm'] == pred_row['Date_file']]

        if len(storms_df) == 0:
            results.append({
                **pred_row.to_dict(),
                'nearest_storm_id': None,
                'has_storm_in_domain': False,
                'is_storm_within_15_deg_radius': False,
                'storm_distance_from_patch_center': None,
            })
        else:
            patch_lat, patch_lon = pred_row['loc']
            patch_center = np.asarray(pred_row['loc']) + np.asarray([15., 15.]) # Due to domain size is 30.
            nearest_storm, distance = _find_nearest_storm(storms_df, patch_center)
            is_in_patch = (patch_lat <= nearest_storm['Lat_storm'] <= patch_lat + 30) and (patch_lon <= nearest_storm['Lon_storm'] <= patch_lon + 30)
            results.append({
                **pred_row.to_dict(),
                'nearest_storm_id': nearest_storm['SID'],
                'has_storm_in_domain': is_in_patch,
                'is_storm_within_15_deg_radius': distance < 15 * 1.414,
                'storm_distance_from_patch_center': distance,
            })


    return pd.DataFrame(results)


ibtracs_df = load_ibtracs('ibtracs.ALL.list.v04r00.csv')
validation_predictions = find_nearest_storm_at_the_same_time(ibtracs_df, validation_predictions)
# -

# True positive samples.

mask = (validation_predictions['y'] == 1) & (validation_predictions['pred_label'] == 1)
true_positives = validation_predictions[mask]
print(f'Number of true positives: ', len(true_positives))
true_positives.head()

# False positives are negative samples that are classified as positive.

mask = (validation_predictions['y'] == 0) & (validation_predictions['pred_label'] == 1)
false_positives = validation_predictions[mask]
print(f'Number of false positives: ', len(false_positives))
false_positives.head()

# False negatives are positive samples that are classified as negative.

mask = (validation_predictions['y'] == 1) & (validation_predictions['pred_label'] == 0)
false_negatives = validation_predictions[mask]
print(f'Number of false negatives: ', len(false_negatives))
false_negatives.head()

# ## Integrated Gradients
# ### Baseline Input
#
# Baseline input is any input that the model return 0. as prediction.
# We will use all zeros input and check if this works.

baseline_input = np.zeros(input_shape, dtype=np.float64)
baseline_pred = model.predict(baseline_input[None, ...])
print(f'{baseline_pred=}')

# Or, another baseline input will be the average of all channels.

# +
preprocessing_layer = model.get_layer(name='preprocessing')
normalization_layer = preprocessing_layer.get_layer(name='normalization')
channel_means = normalization_layer.mean

avg_baseline_input = np.ones(input_shape, dtype=np.float64)[None, ...] * channel_means
avg_baseline_pred = model.predict(avg_baseline_input)
print(f'{avg_baseline_pred=}')
# -

# Seems like that using average is a more suitable option.

# ### Perform Integrated Gradients

avg_baseline_input = tf.cast(tf.convert_to_tensor(avg_baseline_input[0]), dtype=tf.float64)

# +
pressure_levels = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
]
VARIABLES = [
    *[f'absv_{lvl}' for lvl in pressure_levels],
    'capesfc',
    *[f'hgt_{lvl}' for lvl in pressure_levels],
    'pressfc',
    *[f'rh_{lvl}' for lvl in pressure_levels],
    *[f'tmp_{lvl}' for lvl in pressure_levels],
    'tmpsfc',
    *[f'uwind_{lvl}' for lvl in pressure_levels],
    *[f'vwind_{lvl}' for lvl in pressure_levels],
    *[f'wwind_{lvl}' for lvl in pressure_levels],
]

X = tf.convert_to_tensor(false_positives['X'].iloc[0])
igrads = IG.integrated_gradient(model, X, avg_baseline_input).numpy()
print(f'{igrads.shape=}')

# +
fig, axes = plt.subplots(nrows=2, figsize=(24, 12))
ax = axes[0]
positive_attr = np.sum(np.clip(igrads, a_min=0, a_max=None), axis=(0, 1))
ax.plot(range(len(VARIABLES)), positive_attr)
ax.set_xticks(range(len(VARIABLES)))
ax.set_xticklabels(VARIABLES, rotation=45, ha='right')
ax.set_title('Positive Attribution')

ax = axes[1]
negative_attr = np.sum(np.clip(igrads, a_min=None, a_max=0), axis=(0, 1))
ax.plot(range(len(VARIABLES)), negative_attr)
ax.set_xticks(range(len(VARIABLES)))
ax.set_xticklabels(VARIABLES, rotation=45, ha='right')
ax.set_title('Negative Attribution')
fig.tight_layout()


# +
def plot_integrated_gradient(row: pd.Series, *, baseline_input, patch_type: str, domain_size: int = 30):
    print(row)
    ds = xr.load_dataset(row['filename'], engine='netcdf4')
    lat, lon = row['loc']
    ds = ds.sel(lat=slice(lat, lat + domain_size), lon=slice(lon, lon + domain_size))

    visualizer = IntegratedGradientVisualizer()
    X = tf.convert_to_tensor(row['X'])
    igrads = IG.integrated_gradient(model, X, baseline_input).numpy()

    variables_to_plot = [
        ('pressfc', None),
        ('hgtprs', 700),
        ('ugrdprs', 800),
        ('vgrdprs', 800),
        ('capesfc', None),
    ]
    nb_variables = len(variables_to_plot)

    fig = plt.figure(figsize=(25, 15))
    gs = fig.add_gridspec(nrows=3, ncols=5)

    # First, plot the attribution of each variable.
    ax = fig.add_subplot(gs[0, :])
    positive_attr = np.sum(np.clip(igrads, a_min=0, a_max=None), axis=(0, 1))
    ax.plot(range(len(VARIABLES)), positive_attr)
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES, rotation=45, ha='right')
    ax.set_title('Positive Attribution')

    ax = fig.add_subplot(gs[1, :])
    negative_attr = np.sum(np.clip(igrads, a_min=None, a_max=0), axis=(0, 1))
    ax.plot(range(len(VARIABLES)), negative_attr)
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES, rotation=45, ha='right')
    ax.set_title('Negative Attribution')

    for ax_idx, (var, pressure_lvl) in enumerate(variables_to_plot):
        values = ds[var] if pressure_lvl is None else ds[var].sel(lev=pressure_lvl)
        values = values.values

        ax = fig.add_subplot(gs[2, ax_idx])
        cs = ax.pcolormesh(values)
        fig.colorbar(cs, ax=ax)
        ax.set_title(f'{var} at {pressure_lvl}')

        visualizer.visualize(
            integrated_gradients=igrads,
            clip_above_percentile=95,
            clip_below_percentile=28,
            morphological_cleanup=True,
            outlines=False,
            ax=ax,
            use_contour=True,
        )

    fig.suptitle(f'{patch_type}: {row["filename"]} at {lat=}, {lon=}')
    fig.tight_layout()


plot_integrated_gradient(false_positives.iloc[0], patch_type='False Positive', baseline_input=avg_baseline_input)
# -

plot_integrated_gradient(false_positives.iloc[5], patch_type='False Positive', baseline_input=avg_baseline_input)

plot_integrated_gradient(false_positives.iloc[10], patch_type='False Positive', baseline_input=avg_baseline_input)

plot_integrated_gradient(false_positives.iloc[100], patch_type='False Positive', baseline_input=avg_baseline_input)

plot_integrated_gradient(false_positives.iloc[150], patch_type='False Positive', baseline_input=avg_baseline_input)

plot_integrated_gradient(false_positives.iloc[250], patch_type='False Positive', baseline_input=avg_baseline_input)
