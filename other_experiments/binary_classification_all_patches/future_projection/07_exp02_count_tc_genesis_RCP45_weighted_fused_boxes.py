# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %cd ../../..
# %load_ext autoreload
# %autoreload 2
# # %matplotlib widget

from __future__ import annotations
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
from shapely.geometry import Polygon
# -

# In this experiment, I will count the number of TC genesis from
# the patches prediction result.

path = 'other_experiments/binary_classification_all_patches/future_projection/06_exp02_future_projection_RCP45.csv'
df = pd.read_csv(path)
df.head()

# Apply threshold to create genesis prediction.

df['genesis'] = df['pred'] >= 0.5
df.head()

len(df[df['pred'] >= 0.5]) / len(df)

# Groupby 'path' so that we can process all the patches of the same file.

group_df = df.groupby(['path'])
for name, group in group_df:
    print(name, group.head())
    break

# Next, we will count how many TC geneses are detected within the file.

# ## Count Genesis with Spatial and Temporal Merge
#
# If a genesis event is detected,
# the surrounding area is excited and can provide favorable conditions for TC genesis
# for multiple days.
# Thus, in this new way of counting the genesis events,
# we will apply a simple temporal filtering window of 5-7 days to count genesis.
#
# In particular, genesis events are counted as:
# * 

# But first,
# I need to visualize within each patch in each year,
# how many consecutive days that the model detect TCG?

# +
def parse_date(filename: str):
    filename, _ = os.path.splitext(filename)
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, '%Y%m%d_%H_%M')


def plot_tcg_consecutive_days(genesis_pred_df: pd.DataFrame, year: int):
    # Parse date and filter base on year.
    genesis_pred_df['date'] = genesis_pred_df['path'].apply(parse_date)
    genesis_pred_df = genesis_pred_df[genesis_pred_df['date'].apply(lambda d: d.year) == year]

    # Since the rows are sorted, unique days should return sorted days.
    genesis_pred_df = genesis_pred_df.sort_values(['lat', 'lon', 'date'])
    days = list(genesis_pred_df['date'].unique())
    print(days[:10])
    genesis_pred_df['day_nb'] = genesis_pred_df['date'].apply(lambda d: days.index(d))

    # Groupby latitude and longitude.
    genesis_pred_df = genesis_pred_df.groupby(['lat', 'lon'])

    y_labels = []
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (loc, rows) in enumerate(genesis_pred_df):
        genesis_rows = rows[rows['genesis']]
        ax.scatter(genesis_rows['day_nb'], [i] * len(genesis_rows))
        y_labels.append(f'Patch #{i} - ({loc[0]:.0f}, {loc[1]:.0f})')

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(f'Genesis in {year=}')
    fig.tight_layout()


genesis_df = df
plot_tcg_consecutive_days(genesis_df, 2030)
# -

plot_tcg_consecutive_days(genesis_df, 2040)

plot_tcg_consecutive_days(genesis_df, 2050)

# ### Spatial and Temporal Clustering
#
# For this,
# I'll perform a clustering algorithm on each year prediction data to count how many genesis events
# are detected.
# The data to perform the clustering algorithm consists of:
# * Location of the patch (lat & long)
# * Time of the patch.

# +
def create_clustering_data_for_year(genesis_pred_df: pd.DataFrame, year: int):
    # Parse date and filter base on year.
    genesis_pred_df['date'] = genesis_pred_df['path'].apply(parse_date)
    genesis_pred_df = genesis_pred_df[genesis_pred_df['date'].apply(lambda d: d.year) == year]

    # Since the rows are sorted, unique days should return sorted days.
    genesis_pred_df = genesis_pred_df.sort_values(['lat', 'lon', 'date'])
    days = list(genesis_pred_df['date'].unique())
    genesis_pred_df['days_since_May_1st'] = genesis_pred_df['date'].apply(lambda d: days.index(d))
    genesis_pred_df['days_scaled'] = genesis_pred_df['days_since_May_1st'] * 1.5

    # Also, we only care about genesis events.
    return genesis_pred_df[genesis_pred_df['genesis']]


cluster_2030_data_df = create_clustering_data_for_year(genesis_df, 2040)
cluster_2030_data_df.head(10)
# -

# Great,
# now we can use this data to perform clustering.

# +
import sklearn.cluster as cluster # noqa
import numpy as np # noqa

# For now, maybe we should count the clusters manually,
# and then see how the algorithm perform.
#
# for the year 2030, I counted ~20 genesis events.

# kmeans = cluster.KMeans(n_clusters=20)
# kmeans_cluster = kmeans.fit_predict(cluster_2030_data_df[['lat', 'lon', 'days_scaled']])
# print(kmeans_cluster[:5], set(kmeans_cluster))

# +
import seaborn as sns # noqa


def visualize_clusters(cluster_data_df: pd.DataFrame, clusters: np.ndarray, year: int, method: str):
    fig = plt.figure(figsize=(12, 12))

    cluster_df = cluster_data_df.copy()
    cluster_df['cluster'] = clusters

    # 3D plot.
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(
        cluster_df['lon'],
        cluster_df['days_since_May_1st'],
        zs=cluster_df['lat'],
        c=cluster_df['cluster'],
        cmap='tab20c')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Days Since May 1st')
    ax.set_zlabel('Latitude')
    ax.set_zticks([5, 10, 15, 20, 25, 30])

    # Longitudes vs days.
    ax = fig.add_subplot(222)
    sns.stripplot(cluster_df, x='days_since_May_1st', y='lon', hue='cluster', ax=ax, palette='tab20c', jitter=True)
    ax.set_xlabel('Days since May 1st')
    ax.set_ylabel('Longitude')

    # Latitudes vs days.
    ax = fig.add_subplot(223)
    sns.stripplot(cluster_df, x='days_since_May_1st', y='lat', hue='cluster', ax=ax, palette='tab20c', jitter=True)
    ax.set_xlabel('Days since May 1st')
    ax.set_ylabel('Latitude')

    # Patches vs days.
    group_cluster_df = cluster_df.groupby(['lat', 'lon'])
    y_labels = []
    ax = fig.add_subplot(224)
    for i, (loc, rows) in enumerate(group_cluster_df):
        ax.scatter(rows['days_since_May_1st'], [i] * len(rows), c=rows['cluster'], cmap='tab20c')
        y_labels.append(f'Patch #{i} - ({loc[0]:.0f}, {loc[1]:.0f})')

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Days since May 1st')

    fig.suptitle(f'Genesis Clustering in {year=} using {method=}')
    fig.tight_layout()


def visualize_clusters_1plot(cluster_data_df: pd.DataFrame, clusters: np.ndarray, year: int, method: str):
    cluster_data_df = cluster_data_df.copy()
    cluster_data_df['cluster'] = clusters
    unique_clusters = np.unique(clusters)

    cluster_data_df = cluster_data_df.groupby(['lat', 'lon'])
    patches = []
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=0.5)
    for i, (loc, rows) in enumerate(cluster_data_df):
        patches.append(f'Patch #{i} ({loc[0]:.0f}, {loc[1]:.0f})')
        for c in unique_clusters:
            c_rows = rows[rows['cluster'] == c]
            ax.scatter(
                c_rows['days_since_May_1st'],
                [i] * len(c_rows),
                c=c_rows['cluster'],
                marker=f'${c}$',
                cmap='tab20b',
                vmin=unique_clusters.min(),
                vmax=unique_clusters.max())

    ax.set_xlabel('Days since May 1st')
    ax.set_ylabel('Patches')
    ax.set_yticks(range(len(patches)))
    ax.set_yticklabels(patches)
    ax.set_title(f'Cluster for year {year} using {method=}')
    fig.tight_layout()


# +
from ensemble_boxes import * # noqa
import statistics # noqa


def construct_3d_spatial_temporal_genesis_box(genesis_df_with_date: pd.DataFrame):
    def create_current_genesis_box(loc, row: pd.Series):
        return {
            'lat': loc[0],
            'lon': loc[1],
            'from': row['days_since_May_1st'],
            'to': row['days_since_May_1st'] + 1,
            'pred': [row['pred']],
            'date': [row['date']],
        }


    def update_current_genesis_box(cur_box, row: pd.Series):
        assert cur_box['to'] == row['days_since_May_1st']
        cur_box = {**cur_box}
        cur_box['to'] += 1
        cur_box['pred'].append(row['pred'])
        cur_box['date'].append(row['date'])
        return cur_box

    genesis_df = genesis_df_with_date.groupby(['lat', 'lon'])

    boxes = []
    for loc, rows in genesis_df:
        # The rows are ordered in ascending `days_since_May_1st`.
        cur_box = None
        rows = rows[rows['genesis']]
        for _, row in rows.iterrows():
            if cur_box is None:
                # First row.
                cur_box = create_current_genesis_box(loc, row)
            else:
                try:
                    cur_box = update_current_genesis_box(cur_box, row)
                except AssertionError:
                    # The date of the current box the not match with the current date.
                    boxes.append(cur_box)

                    # Create new box.
                    cur_box = create_current_genesis_box(loc, row)

    return pd.DataFrame(boxes)


def generate_box_coordinates(genesis_year_df: pd.DataFrame):
    def scale_min_max(v, min_v, max_v):
        return (v - min_v) / (max_v - min_v)

    lat = genesis_year_df['lat'].values
    lon = genesis_year_df['lon'].values
    lat_min, lat_max = lat.min(), lat.max() + 30
    lon_min, lon_max = lon.min(), lon.max() + 30
    days_min, days_max = genesis_year_df['from'].values.min(), genesis_year_df['to'].values.max()

    genesis_year_df = genesis_year_df.copy()
    genesis_year_df['norm_lat_lower'] = genesis_year_df['lat'].apply(
        lambda l: scale_min_max(l, lat_min, lat_max))
    genesis_year_df['norm_lon_left'] = genesis_year_df['lon'].apply(
        lambda l: scale_min_max(l, lon_min, lon_max))
    genesis_year_df['norm_lat_upper'] = genesis_year_df['lat'].apply(
        lambda l: scale_min_max(l + 30, lat_min, lat_max))
    genesis_year_df['norm_lon_right'] = genesis_year_df['lon'].apply(
        lambda l: scale_min_max(l + 30, lon_min, lon_max))
    genesis_year_df['norm_from'] = genesis_year_df['from'].apply(
        lambda d: scale_min_max(d, days_min, days_max))
    genesis_year_df['norm_to'] = genesis_year_df['to'].apply(
        lambda d: scale_min_max(d, days_min, days_max))
    return genesis_year_df


def merge_spatial_temporal_genesis_box(genesis_year_df: pd.DataFrame, iou_threshold: float | None = None, skip_box_threshold: float = 0.0):
    # Normalize data.
    genesis_year_df = generate_box_coordinates(genesis_year_df)

    # Use the weighted 3d box.
    boxes_list = []
    scores_list = []
    labels_list = []
    for _, row in genesis_year_df.iterrows():
        boxes_list.append([
            row['norm_lat_lower'],
            row['norm_lon_left'],
            row['norm_from'],
            row['norm_lat_upper'],
            row['norm_lon_right'],
            row['norm_to'],
        ])
        scores_list.append(statistics.mean(row['pred']))
        labels_list.append(1.)
    
    # Construct the thingy.
    results = weighted_boxes_fusion_3d(
        [boxes_list],
        [scores_list],
        [labels_list],
        weights=None,
        iou_thr=iou_threshold if iou_threshold is not None else 0.01,
        skip_box_thr=skip_box_threshold)
    return results


cluster_2030_data_df = construct_3d_spatial_temporal_genesis_box(cluster_2030_data_df)
boxes, preds, _ = merge_spatial_temporal_genesis_box(cluster_2030_data_df, iou_threshold=0.4)
(preds > 0.6).sum(), len(preds)


# +
def count_genesis_in_each_year(genesis_df: pd.DataFrame):
    # We have these years.
    years = list(range(2030, 2051)) + list(range(2080, 2101))

    genesis_counts = []
    for year in years:
        cluster_data_df = create_clustering_data_for_year(genesis_df, year)
        cluster_data_df = construct_3d_spatial_temporal_genesis_box(cluster_data_df)
        boxes, preds, _ = merge_spatial_temporal_genesis_box(cluster_data_df, iou_threshold=0.4)
        genesis_counts.append({
            'year': year,
            'genesis': (preds > 0.6).sum(),
        })

    return pd.DataFrame(genesis_counts)


genesis_count_df = count_genesis_in_each_year(genesis_df)
fig, axes = plt.subplots(nrows=2, figsize=(18, 6))
ax = axes[0]
df = genesis_count_df[genesis_count_df['year'] <= 2050]
sns.lineplot(df, x='year', y='genesis', ax=ax)
ax.set_title(f'Genesis count for year 2030-2050')
ax.set_xlabel('Year')
ax.set_ylabel('Genesis Count')

ax = axes[1]
df = genesis_count_df[genesis_count_df['year'] > 2050]
sns.lineplot(df, x='year', y='genesis', ax=ax)
ax.set_title(f'Genesis count for year 2080-2100')
ax.set_xlabel('Year')
ax.set_ylabel('Genesis Count')
fig.tight_layout()
# -

fig, ax = plt.subplots(figsize=(18, 6))
df = genesis_count_df[genesis_count_df['year'] <= 2050]
nb_years = len(df)
ax.plot(range(nb_years), df['genesis'], label='2030-2050')
df = genesis_count_df[genesis_count_df['year'] > 2050]
ax.plot(range(nb_years), df['genesis'], label='2080-2100')
ax.set_xticks(range(nb_years))
ax.set_xticklabels([f'{2030 + i}\n{2080 + i}' for i in range(nb_years)])
ax.set_ylabel('Genesis Count')
ax.set_xlabel('Year')
ax.legend()
fig.tight_layout()
