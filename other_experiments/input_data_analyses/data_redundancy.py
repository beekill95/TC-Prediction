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

# +
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xarray as xr
# -

# # PCA on Individual Reanalysis Data
#
# This is not what we want since we're forecasting
# across different observations.
# Then, we should care about how these variables change overtime,
# not about changes within an observation.
#
# ## Northern Pacific Ocean

data_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/fnl_20080501_00_00.nc'
dataset = xr.load_dataset(data_path)
dataset

# ### All Features PCA
# First, let us transform the xarray dataset into pandas dataframe.

# +
def convert_dataset_to_pd(ds: xr.Dataset) -> pd.DataFrame:
    levels = ds['lev']

    df = dict()
    for variable in ds.data_vars:
        var_values = ds[variable]

        if len(var_values.dims) == 3:
            for lev in levels.values:
                values = var_values.sel(lev=lev).values.flatten()
                df[f'{variable}_{lev}'] = values
        else:
            values = var_values.values.flatten()
            df[f'{variable}'] = values

    return pd.DataFrame(df)


df = convert_dataset_to_pd(dataset)
df.head()
# -

# Now, we can perform PCA on that dataset.

# +
def pca_dataframe(df: pd.DataFrame):
    normalized_df = (df - df.mean()) / df.std()

    pca = PCA(n_components=df.shape[1])
    pca.fit(normalized_df)

    return pca


def eigenvalues_dataframe(df: pd.DataFrame):
    normalized_df = (df - df.mean()) / df.std()
    C = normalized_df.cov()
    
    eigenvalues, _ = np.linalg.eig(C.values)
    return pd.DataFrame(dict(Feature=df.columns, Eigenvalue=eigenvalues))

pca = pca_dataframe(df)
eig = eigenvalues_dataframe(df)
# -

plt.plot(pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylabel('Cummulative Explained Variance')
plt.xlabel('Components')
plt.show()

plt.figure(figsize=(16, 8))
plt.bar(range(40), eig['Eigenvalue'][:40])
plt.xticks(range(40), df.columns[:40], rotation=45, ha='right')
plt.tight_layout()
