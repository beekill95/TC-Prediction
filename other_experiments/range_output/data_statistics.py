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

# %cd ../..

from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Data Statistics For Time Range Outputs

label_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_time_range_24h_with_0h_no_genesis_class.csv'
label_df = pd.read_csv(
        label_path,
        converters=dict(
            Genesis=literal_eval,
            Genesis_Location=literal_eval,
            Genesis_SID=literal_eval,
            Other_TC=literal_eval))
label_df.head(10)

# ## Negative vs Positive Labels

genesis_cnt = label_df['Genesis'].apply(lambda gt: sum(gt))
has_genesis = genesis_cnt > 0
print('Number of records with genesis: ', has_genesis.sum(),
      '\nNumber of records: ', len(genesis_cnt),
      '\nRatio between with/without genesis: ', has_genesis.sum() / (len(genesis_cnt) - has_genesis.sum()))

# ## How many records with multiple genesis?

genesis_cnt.hist()
plt.xticks(range(5))
plt.title('Genesis Count Histogram')
plt.tight_layout()

multiple_genesis = genesis_cnt > 1
print(
    'Number of records with multiple genesis: ', multiple_genesis.sum(),
    '\nCompare with number of records with only 1 genesis: ', (genesis_cnt == 1).sum())

# ## How many records with TCs?

has_tc_cnt = label_df['Other_TC'].apply(lambda tc: len(tc))
has_tc = has_tc_cnt > 0
print('Number of records with TC: ', has_tc.sum(),
      '\nNumber of records without TC: ', (~has_tc).sum(),
      '\nRatio between with and without TC: ', has_tc.sum() / (~has_tc).sum())

# ## Comparison between Genesis and TC
# ### How many records with both Genesis and TC?

has_genesis_and_tc = has_genesis & has_tc
has_genesis_only = has_genesis & ~has_tc
has_tc_only = ~has_genesis & has_tc
print('Number of records with both genesis and TC: ', has_genesis_and_tc.sum(),
      '\nNumber of records with genesis only: ', has_genesis_only.sum(),
      '\nNumber of records with TC only: ', has_tc_only.sum())

# ## Spatial Distribution of Genesis

# +
domain = (41, 161)
genesis_loc_at_0h = label_df['Genesis_Location'].apply(lambda gt: gt[0])
genesis_spatial_cnt = np.zeros(domain)
latitudes = np.arange(5, 46, 1)
longitudes = np.arange(100, 261, 1)
for genesis_loc in genesis_loc_at_0h:
    if not len(genesis_loc):
        continue

    for loc in genesis_loc:
        lat, lon = loc

        lat_idx = np.argmin(np.abs(lat - latitudes))
        lon_idx = np.argmin(np.abs(lon - longitudes))

        genesis_spatial_cnt[lat_idx, lon_idx] += 1

plt.figure(figsize=(12, 4))
plt.title('TC Genesis Spatial Distribution')
# cs = plt.contourf(genesis_spatial_cnt, cmap='Reds', levels=range(int(genesis_spatial_cnt.max()) + 1))
cs = plt.pcolormesh(genesis_spatial_cnt, cmap='Reds')
plt.colorbar(cs)
plt.xticks(range(0, 161, 10), longitudes[::10])
plt.yticks(range(0, 41, 5), latitudes[::5])
plt.tight_layout()
