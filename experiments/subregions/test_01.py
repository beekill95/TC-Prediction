# -*- coding: utf-8 -*-
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
# %load_ext autoreload
# %autoreload 2

import tc_formation.data.subregions as subregions
import tc_formation.data.subregions.data_loader

# # Test implementation of subregions module
# ## Specify data

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
subregion_size = (20, 20) # in degree
subregion_stride = 5 # in degree
data_shape = (41, 161, 13)

# ## Specify data loader

data_loader = subregions.data_loader.SubRegionsTimeSeriesTropicalCycloneDataLoader(
    data_shape=data_shape,
    subset=subset,
    subregion_size=subregion_size,
    subregion_stride=subregion_stride,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=2,
    leadtimes=12,
    shuffle=True,
    negative_subregions_ratio=3,
)

# ### Loop through data to see if we get the correct result.

print('=== Multiple Time Step ===')
training_iter = iter(training)
for X, y in training_iter:
    print(X.shape)
    break

# # Test implementation of Single Time Subregion

data_loader = subregions.data_loader.SubRegionsTropicalCycloneDataLoader(
    data_shape=data_shape,
    subset=subset,
    subregion_size=subregion_size,
    subregion_stride=subregion_stride,
)
training = data_loader.load_dataset(
    train_path,
    batch_size=512,
    shuffle=True,
    negative_subregions_ratio=3,
    caching=False,
)
validation = data_loader.load_dataset(
    train_path,
    batch_size=512,
    shuffle=False,
    negative_subregions_ratio=None,
)

print('=== Single Time Step ===')
i = 0
training_iter = iter(training)
for X, y in training_iter:
    i += 1
    print(i, X.shape, y.shape)


# ## Test Validation without negative subsample

val_iter = iter(validation)
i = 0
for X, y in val_iter:
    i += 1
    print(i, X.shape, y.shape)
