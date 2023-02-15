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
# %cd ../../..
# %load_ext autoreload
# %autoreload 2
# # %matplotlib widget

from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
from shapely.geometry import Polygon
# -

# In this experiment, I will count the number of TC genesis from
# the patches prediction result.

path = 'other_experiments/binary_classification_all_patches/06_exp01_future_projection_RCP45.csv'
df = pd.read_csv(path)
df.head()

# Apply threshold to create genesis prediction.

df['genesis'] = df['pred'] >= 0.6
df.head()

len(df[df['pred'] >= 0.5]) / len(df)

# Groupby 'path' so that we can process all the patches of the same file.

group_df = df.groupby(['path'])
for name, group in group_df:
    print(name, group.head())
    break

# Next, we will count how many TC geneses are detected within the file.

# + tags=[]
def Rectangle(x, y, w, h):
    return Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])


def count_genesis(r: pd.DataFrame, domain_size=30):
    genesis_areas = []
    for _, row in r.iterrows():
        if row['genesis']:
            area = Rectangle(
                row['lon'], row['lat'], domain_size, domain_size)
            overlaped_idx = -1
            for idx, known_area in enumerate(genesis_areas):
                if known_area.overlaps(area):
                    overlaped_idx = idx
                    break

            if overlaped_idx != -1:
                known_area = genesis_areas[overlaped_idx]
                genesis_areas[overlaped_idx] = known_area.union(area)
            else:
                genesis_areas.append(area)

    return len(genesis_areas)


def parse_date(filename: str):
    filename, _ = os.path.splitext(filename)
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, '%Y%m%d_%H_%M')


count_results = []
for name, group in group_df:
    nb_genesis = count_genesis(group)
    count_results.append({
        'path': name,
        'genesis': nb_genesis,
        'date': parse_date(name),
    })
count_results = pd.DataFrame(count_results)
count_results['year'] = count_results['date'].apply(lambda d: d.year)
count_results.head()
# -

# Count number of genesis.
print(f'{len(count_results)=}, {(count_results["genesis"] > 0).sum()=}')

yearly_count = count_results.groupby('year').agg({'genesis': 'sum'})
yearly_count.head()


# +
# count_results.to_csv('genesis_daily_count.csv')
# yearly_count.to_csv('genesis_yearly_count.csv')
# -

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

# + [markdown] tags=[]
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


cluster_2030_data_df = create_clustering_data_for_year(genesis_df, 2030)
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


# visualize_clusters(cluster_2030_data_df, kmeans_cluster, 2030, 'Kmeans')
# -

dbscan = cluster.DBSCAN(eps=6, min_samples=2)
dbscan_cluster = dbscan.fit_predict(cluster_2030_data_df[['lat', 'lon', 'days_scaled']])
print(dbscan_cluster[:5], set(dbscan_cluster))
# visualize_clusters(cluster_2030_data_df, dbscan_cluster, 2030, 'DBSCAN')
visualize_clusters_1plot(cluster_2030_data_df, dbscan_cluster, 2030, 'DBSCAN')


# +
def count_genesis_in_each_year(genesis_df: pd.DataFrame):
    # We have these years.
    years = list(range(2030, 2051)) + list(range(2080, 2101))

    genesis_counts = []
    for year in years:
        cluster_data_df = create_clustering_data_for_year(genesis_df, year)
        dbscan = cluster.DBSCAN(eps=6, min_samples=2)
        dbscan_cluster = dbscan.fit_predict(
            cluster_data_df[['lat', 'lon', 'days_scaled']])
        genesis_counts.append({
            'year': year,
            'genesis': len(np.unique(dbscan_cluster[dbscan_cluster >= 0])),
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
ax.set_title(f'Genesis count for year 2050-2080')
ax.set_xlabel('Year')
ax.set_ylabel('Genesis Count')
fig.tight_layout()
