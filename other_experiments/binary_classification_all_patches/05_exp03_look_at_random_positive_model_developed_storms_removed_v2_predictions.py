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
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import xarray as xr


# +
val_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_developed_storms_removed_v2_Val.tfrecords'
test_path = 'data/ncep_WP_EP_6h_all_binary_patches_all_variables_developed_storms_removed_v2_Test.tfrecords'

dataloader = PatchesWithGenesisTFRecordDataLoader()
val_patches_ds = dataloader.load_dataset(val_path, batch_size=256, for_analyzing=True)
test_patches_ds = dataloader.load_dataset(test_path, batch_size=256)

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
test_patches_ds = test_patches_ds.map(set_shape(input_shape))


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


path = 'saved_models/random_positive_no_pca_developed_storms_removed_v2_f1_0.628'
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

# Check that the model results is the same with what we got.
model.evaluate(test_patches_ds)

# ## Visualize Feature Maps

# +
from sklearn.manifold import TSNE # noqa
import seaborn as sns # noqa

feature_map = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name='flatten').output,
)

val_embedding = []
val_label = []
val_label_pred = []
for X, y, *_ in iter(val_patches_ds):
    X_emb = feature_map(X)
    y_pred = model(X)
    val_embedding.append(X_emb)
    val_label.append(y)
    val_label_pred.append(y_pred)

val_embedding = np.concatenate(val_embedding, axis=0)
val_label = np.concatenate(val_label, axis=0)
val_label_pred = np.concatenate(val_label_pred, axis=0)

tsne = TSNE()
val_embedding_2 = tsne.fit_transform(val_embedding)
df = pd.DataFrame({
    'f1': val_embedding_2[:, 0],
    'f2': val_embedding_2[:, 1],
    'y': val_label[:, 0],
    'y_pred': np.where(val_label_pred[:, 0] > 0.5, 1, 0),
})
fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
ax = axes[0]
sns.scatterplot(df, x='f1', y='f2', hue='y', style='y', ax=ax)
ax.set_title('Feature Map Conv Visualization with true label')
ax = axes[1]
sns.scatterplot(df, x='f1', y='f2', hue='y_pred', style='y_pred', ax=ax)
ax.set_title('Feature Map Conv Visualization with predicted label')

# +
feature_map_dense = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(name='dense').output,
)

val_embedding = []
val_label = []
val_label_pred = []
for X, y, *_ in iter(val_patches_ds):
    X_emb = feature_map_dense(X)
    y_pred = model(X)
    val_embedding.append(X_emb)
    val_label.append(y)
    val_label_pred.append(y_pred)

val_embedding = np.concatenate(val_embedding, axis=0)
val_label = np.concatenate(val_label, axis=0)
val_label_pred = np.concatenate(val_label_pred, axis=0)

tsne = TSNE()
val_embedding_2 = tsne.fit_transform(val_embedding)
df = pd.DataFrame({
    'f1': val_embedding_2[:, 0],
    'f2': val_embedding_2[:, 1],
    'y': val_label[:, 0],
    'y_pred': np.where(val_label_pred[:, 0] > 0.5, 1, 0),
})
fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
ax = axes[0]
sns.scatterplot(df, x='f1', y='f2', hue='y', style='y', ax=ax)
ax.set_title('Feature Map Dense Visualization with true label')
ax = axes[1]
sns.scatterplot(df, x='f1', y='f2', hue='y_pred', style='y_pred', ax=ax)
ax.set_title('Feature Map Dense Visualization with predicted label')
# -

# ## Look at Model's Predictions
#
# In this section,
# I will run the model to predict samples on the validation dataset,
# to see:
#
# * Why positive samples are classified as False?
# * Why negative samples are classified as True?
# * Is there any systematic bias that the model is making?

validation_predictions = []
for X, y, filenames, patch_locations in tqdm(iter(val_patches_ds), desc='Predicting on Validation'):
    y_pred = model.predict(X, verbose=False)
    validation_predictions.extend(
        [dict(filename=filename.numpy().decode('utf-8'), loc=loc.numpy(), y=yi.numpy()[0], pred=yi_pred[0])
         for filename, loc, yi, yi_pred in zip(filenames, patch_locations, y, y_pred)])

pd.set_option('display.max_colwidth', None)
validation_predictions = pd.DataFrame(validation_predictions)
validation_predictions['pred_label'] = validation_predictions['pred'].apply(lambda p: 1 if p > 0.5 else 0)
validation_predictions.head()

# ### False Positives
#
# False positives are negative samples but are classified as positive.

mask = (validation_predictions['y'] == 0) & (validation_predictions['pred_label'] == 1)
false_positives = validation_predictions[mask]
print(f'Number of false positives: ', len(false_positives))
false_positives.head()

# #### Statistics

# +
def show_location_histogram(df: pd.DataFrame):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    ax = axes[0]
    lat = df['loc'].apply(lambda loc: loc[0])
    lat.hist(ax=ax)
    ax.set_title('Latitudes Histogram')

    ax = axes[1]
    lon = df['loc'].apply(lambda loc: loc[1])
    lon.hist(ax=ax)
    ax.set_title('Longitudes Histogram')

    fig.tight_layout()

    
print('Number of file having false positives:', len(false_positives.groupby('filename')))
show_location_histogram(false_positives)
# -

# How many of these patches have a developed tropical cyclone inside them?

# +
from datetime import datetime # noqa
import os # noqa


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
false_positives_with_nearest_storm = find_nearest_storm_at_the_same_time(ibtracs_df, false_positives)
false_positives_with_nearest_storm.head()
# -

false_positives_with_nearest_storm

# OK, now we're ready to count in these false positive patches,
# how many patches contain a storm in it.

nb_false_positives_has_storm = false_positives_with_nearest_storm['has_storm_in_domain'].sum()
print(
    'Number of false positive patches has a storm in it:',
    nb_false_positives_has_storm,
    '\nAnd the number of false positive patches is:',
    len(false_positives_with_nearest_storm),
    '\nThe ratio is:',
    nb_false_positives_has_storm / len(false_positives_with_nearest_storm))

# There are almost 50% of the false positive patches having a storm in it.
# I want to know as well how many patches have a storm near it (within radius of $15\sqrt{2}$deg).

nb_false_positives_has_storm_within_15deg = false_positives_with_nearest_storm['is_storm_within_15_deg_radius'].sum()
print(
    'Number of false positive patches has a storm within 15deg:',
    nb_false_positives_has_storm_within_15deg,
    '\nAnd the number of false positive patches is:',
    len(false_positives_with_nearest_storm),
    '\nThe ratio is:',
    nb_false_positives_has_storm_within_15deg / len(false_positives_with_nearest_storm))

# Let's see the remaining patches, how are they look like.

false_positives_without_storm_in_domain = false_positives_with_nearest_storm[~false_positives_with_nearest_storm['is_storm_within_15_deg_radius']]
false_positives_without_storm_in_domain.head(10)

# In addition, I want to know the distribution of the predicted probability for these patches.

# +
def show_predicted_probability_historam_of_false_positives_with_and_without_developed_storms(false_positives_with_nearest_storm: pd.DataFrame):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)

    # Histogram of the predicted probability with developed storms.
    ax = axes[0]
    with_storms = false_positives_with_nearest_storm[false_positives_with_nearest_storm['has_storm_in_domain']]
    with_storms['pred'].hist(ax=ax)
    ax.set_title('Histogram with developed storms')

    # Histogram of the predicted probability without developed storms.
    ax = axes[1]
    without_storms = false_positives_with_nearest_storm[~false_positives_with_nearest_storm['has_storm_in_domain']]
    without_storms['pred'].hist(ax=ax)
    ax.set_title('Histogram without developed storms')

    fig.tight_layout()


show_predicted_probability_historam_of_false_positives_with_and_without_developed_storms(false_positives_with_nearest_storm)
# -

# #### Some False Positive Patches
#
# Is there anything common within these negative patches that cause the model to classify them as positive?

# +
def plot_patch(row: pd.Series, patch_type: str, domain_size: int = 30):
    ds = xr.load_dataset(row['filename'], engine='netcdf4')
    lat, lon = row['loc']
    ds = ds.sel(lat=slice(lat, lat + domain_size), lon=slice(lon, lon + domain_size))

    variables_to_plot = [
        ('pressfc', None),
        ('ugrdprs', 800),
        ('vgrdprs', 800),
        ('capesfc', None),
    ]
    nb_variables = len(variables_to_plot)
    fig, axes = plt.subplots(ncols=nb_variables, figsize=(4 * nb_variables, 4))
    for ax, (var, pressure_lvl) in zip(axes, variables_to_plot):
        values = ds[var] if pressure_lvl is None else ds[var].sel(lev=pressure_lvl)
        values = values.values

        cs = ax.pcolormesh(values)
        fig.colorbar(cs, ax=ax)
        ax.set_title(f'{var} at {pressure_lvl}')

    fig.suptitle(f'{patch_type}: {row["filename"]} at {lat=}, {lon=}')
    fig.tight_layout()


plot_patch(false_positives.iloc[0], 'False Positive')
# -

plot_patch(false_positives.iloc[1], 'False Positive')

plot_patch(false_positives.iloc[2], 'False Positive')

plot_patch(false_positives.iloc[50], 'False Positive')

# ### False Negatives
#
# False negatives are positive samples but are classified as negative.

mask = (validation_predictions['y'] == 1) & (validation_predictions['pred_label'] == 0)
false_negatives = validation_predictions[mask]
print(f'Number of false positives: ', len(false_negatives))
false_negatives.head()

print('Number of file having false negatives:', len(false_negatives.groupby('filename')))
show_location_histogram(false_negatives)

false_negatives['pred'].hist()

# Do these false negative cases even have a developed storm in it?

false_negatives_with_nearest_storm = find_nearest_storm_at_the_same_time(ibtracs_df, false_negatives)
false_negatives_with_nearest_storm.head()

nb_false_negatives_has_storm = false_negatives_with_nearest_storm['has_storm_in_domain'].sum()
print(
    'Number of false negative patches has a storm in it:',
    nb_false_negatives_has_storm,
    '\nAnd the number of false negative patches is:',
    len(false_negatives_with_nearest_storm),
    '\nThe ratio is:',
    nb_false_negatives_has_storm / len(false_negatives_with_nearest_storm))

# For these false negative cases,
# I want to know where the genesis location is.

# +
def find_genesis_location(pred_df: pd.DataFrame, ibtracs_df: pd.DataFrame, domain_size=30):
    def filter_not_in_domain(row: pd.Series):
        lat, lon = row['loc']
        return (lat <= row['Lat_storm'] <= lat + domain_size) and (lon <= row['Lon_storm'] <= lon + domain_size)

    pred_df['Date_file'] = pred_df['filename'].apply(parse_date_from_nc_filename)

    # It's better to shrink our domain,
    # since we know that we only detect storms in WP and EP.
    ibtracs_df = ibtracs_df.groupby('SID').first()
    ibtracs_df = ibtracs_df[ibtracs_df['BASIN'].isin(['WP', 'EP'])]
    ibtracs_df['Date_storm_leadtime'] = ibtracs_df['Date_storm'].apply(lambda d: d - timedelta(hours=6))
    # Cheated a bit.
    lon_mask = (100 <= ibtracs_df['Lon_storm']) & (ibtracs_df['Lon_storm'] <= 260)
    lat_mask = (5 <= ibtracs_df['Lat_storm']) & (ibtracs_df['Lat_storm'] <= 45)
    ibtracs_df = ibtracs_df[lon_mask & lat_mask]

    results = pred_df.merge(ibtracs_df, left_on='Date_file', right_on='Date_storm_leadtime')
    results = results[results.apply(filter_not_in_domain, axis=1)]

    return results

false_negatives_with_genesis_location = find_genesis_location(false_negatives, ibtracs_df)
print(len(false_negatives_with_genesis_location), len(false_negatives))
false_negatives_with_genesis_location.head()

# +
def plot_histogram_of_genesis_locations_wrt_domain_edges(pred_with_genesis_loc_df: pd.DataFrame, domain_size=30):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    g_lat = pred_with_genesis_loc_df['Lat_storm']
    g_lon = pred_with_genesis_loc_df['Lon_storm']
    patch_loc = pred_with_genesis_loc_df['loc']

    # Genesis distance from left edge.
    ax = axes[0, 0]
    distance_from_left_edge = g_lon - patch_loc.apply(lambda l: l[1])
    distance_from_left_edge.hist(ax=ax, bins=domain_size)
    ax.set_title('Genesis distance from left edge')
    print('Number of cases with distance from left edge < 5deg:', (distance_from_left_edge < 5).sum())

    # Genesis distance from right edge.
    ax = axes[0, 1]
    distance_from_right_edge = patch_loc.apply(lambda l: l[1] + domain_size) - g_lon
    distance_from_right_edge.hist(ax=ax, bins=domain_size)
    ax.set_title('Genesis distance from right edge')
    print('Number of cases with distance from right edge < 5deg:', (distance_from_right_edge < 5).sum())

    # Genesis distance from top edge.
    ax = axes[1, 0]
    distance_from_top_edge = patch_loc.apply(lambda l: l[0] + domain_size) - g_lat
    distance_from_top_edge.hist(ax=ax, bins=domain_size)
    ax.set_title('Genesis distance from top edge')

    # Genesis distance from bottom edge.
    ax = axes[1, 1]
    distance_from_bottom_edge = g_lat - patch_loc.apply(lambda l: l[0])
    distance_from_bottom_edge.hist(ax=ax, bins=domain_size)
    ax.set_title('Genesis distance from bottom edge')
    print('Number of cases with distance from bottom edge < 5deg:', (distance_from_bottom_edge < 5).sum())

    fig.tight_layout()


plot_histogram_of_genesis_locations_wrt_domain_edges(false_negatives_with_genesis_location)
# -

# So in actual, how many false negative cases with genesis that are too close the domain edges?

# +
from functools import reduce # noqa


def count_cases_with_genesis_close_to_domain_edges(pred_with_genesis_loc_df: pd.DataFrame, domain_size=30, threshold=5):
    g_lat = pred_with_genesis_loc_df['Lat_storm']
    g_lon = pred_with_genesis_loc_df['Lon_storm']
    patch_loc = pred_with_genesis_loc_df['loc']

    distance_from_left_edge = g_lon - patch_loc.apply(lambda l: l[1])
    distance_from_right_edge = patch_loc.apply(lambda l: l[1] + domain_size) - g_lon
    distance_from_top_edge = patch_loc.apply(lambda l: l[0] + domain_size) - g_lat
    distance_from_bottom_edge = g_lat - patch_loc.apply(lambda l: l[0])

    near_edges = reduce(lambda acc, cur: acc | (cur < threshold), [
        distance_from_left_edge,
        distance_from_right_edge,
        distance_from_bottom_edge,
    ], distance_from_top_edge < threshold)
    nb_cases = near_edges.sum()

    print('Number of cases near edges:', nb_cases,
          '\nOver total number of cases:', len(pred_with_genesis_loc_df),
          '\nRatio:', nb_cases / len(pred_with_genesis_loc_df))


count_cases_with_genesis_close_to_domain_edges(false_negatives_with_genesis_location)
# -

# But wait, is this the same as in the correct cases?
# So my hypothesis is that: because the genesis events are too close to the edges,
# the model is having a hard time detecting genesis event.
#
# Thus, the null hypothesis must be true right?
# The null hypothesis is that: the model will easily detect genesis events if the
# genesis is near the middle of the patch.
#
# Therefore, the expectation is that:
# when we calculate the number correctly classified genesis events,
# the ratio of near edges must be low.

mask = (validation_predictions['y'] == 1) & (validation_predictions['pred_label'] == 1)
true_positives = validation_predictions[mask]
true_positives_with_genesis_location = find_genesis_location(true_positives, ibtracs_df)
print(len(true_positives), len(true_positives_with_genesis_location))
count_cases_with_genesis_close_to_domain_edges(true_positives_with_genesis_location)
plot_histogram_of_genesis_locations_wrt_domain_edges(true_positives_with_genesis_location)

# Among these correctly classified samples,
# does the model have a hard time making the prediction?
#
# If it does, then the prediction of the near edges cases should be near 0.5,
# while the remaining cases should not.

# +
def plot_histogram_of_near_edges_prediction(pred_with_genesis_loc_df: pd.DataFrame, domain_size=30, threshold=5):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    # TODO: should be enclosed in a function.
    g_lat = pred_with_genesis_loc_df['Lat_storm']
    g_lon = pred_with_genesis_loc_df['Lon_storm']
    patch_loc = pred_with_genesis_loc_df['loc']

    distance_from_left_edge = g_lon - patch_loc.apply(lambda l: l[1])
    distance_from_right_edge = patch_loc.apply(lambda l: l[1] + domain_size) - g_lon
    distance_from_top_edge = patch_loc.apply(lambda l: l[0] + domain_size) - g_lat
    distance_from_bottom_edge = g_lat - patch_loc.apply(lambda l: l[0])

    near_edges_mask = reduce(lambda acc, cur: acc | (cur < threshold), [
        distance_from_left_edge,
        distance_from_right_edge,
        distance_from_bottom_edge,
    ], distance_from_top_edge < threshold)


    # Near edges predicted probability histogram.
    ax = axes[0]
    near_edges = pred_with_genesis_loc_df[near_edges_mask]
    near_edges['pred'].hist(ax=ax)
    ax.set_title('Near edges predicted probability histogram.')

    # Non near edges predicted probability histogram.
    ax = axes[1]
    non_near_edges = pred_with_genesis_loc_df[~near_edges_mask]
    non_near_edges['pred'].hist(ax=ax)
    ax.set_title('Non Near edges predicted probability histogram.')

    fig.tight_layout()


plot_histogram_of_near_edges_prediction(true_positives_with_genesis_location)
# -

# Well, my hypothesis is not correct.
# I want to know how important the genesis near edges is?
# i.e. How many positive cases that has genesis near the edges?

positive_cases = validation_predictions[validation_predictions['y'] == 1]
positive_cases_with_genesis_location = find_genesis_location(positive_cases, ibtracs_df)
count_cases_with_genesis_close_to_domain_edges(positive_cases_with_genesis_location)

# ## Actions
