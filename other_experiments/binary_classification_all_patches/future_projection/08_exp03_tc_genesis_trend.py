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
import arviz as az
import jax.numpy as jnp
import jax.random as random
from datetime import datetime
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import NUTS, MCMC
import os
import pandas as pd
from tc_formation.tcg_analysis.clustering import DBScanClustering, WeightedFusedBoxesClustering


numpyro.set_host_device_count(4)
# -

# # TC Trend on RCP45 and RCP85
#
# In this experiment,
# I will count the number of TC genesis from the patches prediction result in two scenarios,
# using DBScan and Bayesian analysis.

# +
def parse_date(filename: str):
    filename, _ = os.path.splitext(filename)
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, '%Y%m%d_%H_%M')


path_rcp45 = 'other_experiments/binary_classification_all_patches/future_projection/06_exp02_future_projection_RCP45.csv'
rcp45_df = pd.read_csv(path_rcp45)
rcp45_df['date'] = rcp45_df['path'].apply(parse_date)
rcp45_df.head()
# -


path_rcp85 = 'other_experiments/binary_classification_all_patches/future_projection/06_exp02_future_projection_RCP85.csv'
rcp85_df = pd.read_csv(path_rcp85)
rcp85_df['date'] = rcp85_df['path'].apply(parse_date)
rcp85_df.head()

# ## Genesis Count
# ### DBScan

clustering = DBScanClustering(0.6)
rcp45_count_df = clustering.count_genesis(rcp45_df)
rcp85_count_df = clustering.count_genesis(rcp85_df)
rcp45_count_df['rcp'] = 0
rcp85_count_df['rcp'] = 1
dbscan_count_df = pd.concat([rcp45_count_df, rcp85_count_df])
dbscan_count_df['cluster'] = 'dbscan'

# ### Weighted Fused Boxes

# +
from tc_formation.tcg_analysis.clustering import WeightedFusedBoxesClustering

clustering = WeightedFusedBoxesClustering(iou_threshold=0.4)
rcp45_count_df = clustering.count_genesis(rcp45_df)
rcp85_count_df = clustering.count_genesis(rcp85_df)
rcp45_count_df['rcp'] = 0
rcp85_count_df['rcp'] = 1
wfb_count_df = pd.concat([rcp45_count_df, rcp85_count_df])
wfb_count_df['cluster'] = 'wfb'
# -

# ## Trend Analysis

# +
from tc_formation.tcg_analysis.models import hier_tcg_trend_year_rcp_cluster_model # noqa


count_df = pd.concat([dbscan_count_df, wfb_count_df])
count_df['cluster'] = count_df['cluster'].astype('category')

kernel = NUTS(hier_tcg_trend_year_rcp_cluster_model, target_accept_prob=0.999)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    tcg_freq=jnp.array(count_df['genesis'].values),
    year=jnp.array(count_df['year'].values),
    rcp=jnp.array(count_df['rcp'].values),
    cluster=jnp.array(count_df['cluster'].cat.codes.values),
)
mcmc.print_summary()
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(rcp=['RCP45', 'RCP85'], cluster=count_df['cluster'].cat.categories),
    dims=dict(b2=['rcp'], b3=['cluster']))
az.plot_trace(idata)
plt.tight_layout()

# ### Year

az.plot_posterior(idata, var_names=['b1'], hdi_prob=0.95, ref_val=0., point_estimate='mode')

# ### RCP Scenarios

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP45'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP85'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

# ### Clustering Method

az.plot_posterior(idata, var_names=['b3'], coords=dict(cluster='dbscan'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b3'], coords=dict(cluster='wfb'), hdi_prob=0.95, ref_val=0., point_estimate='mode')
