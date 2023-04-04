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
from tc_formation.tcg_analysis.clustering import WeightedFusedBoxesClustering


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

# +
from tc_formation.tcg_analysis.clustering import WeightedFusedBoxesClustering

clustering = WeightedFusedBoxesClustering(iou_threshold=0.4)
rcp45_count_df = clustering.count_genesis(rcp45_df)
rcp45_count_df.head()
# -

rcp85_count_df = clustering.count_genesis(rcp85_df)
rcp85_count_df.head()

# Now, we can display the genesis trend.

# +
def display_genesis_trend(count_df: pd.DataFrame):
    _, ax = plt.subplots(figsize=(18, 6), layout='constrained')

    mid_century_df = count_df[count_df['year'] <= 2050]
    end_century_df = count_df[count_df['year'] > 2050]

    nb_years = len(mid_century_df)
    ax.plot(range(nb_years), mid_century_df['genesis'], label='2030-2050')
    ax.plot(range(nb_years), end_century_df['genesis'], label='2080-2100')
    ax.legend()
    ax.set_xticks(range(nb_years))
    ax.set_xticklabels(f'{2030 + i}\n{2080 + i}' for i in range(nb_years))
    ax.set_xlabel('Year')
    ax.set_ylabel('Genesis Frequency')


display_genesis_trend(rcp45_count_df)
# -

display_genesis_trend(rcp85_count_df)

# ## Trend Analysis

# +
from tc_formation.tcg_analysis.models import hier_tcg_trend_year_rcp_model # noqa


rcp45_count_df['rcp'] = 0
rcp85_count_df['rcp'] = 1
count_df = pd.concat([rcp45_count_df, rcp85_count_df])

kernel = NUTS(hier_tcg_trend_year_rcp_model, target_accept_prob=0.999)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    tcg_freq=jnp.array(count_df['genesis'].values),
    year=jnp.array(count_df['year'].values),
    rcp=jnp.array(count_df['rcp'].values),
)
mcmc.print_summary()
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(rcp=['RCP45', 'RCP85']),
    dims=dict(b2=['rcp']))
az.plot_trace(idata)
plt.tight_layout()

az.plot_posterior(idata, var_names=['b1'], hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP45'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP85'), hdi_prob=0.95, ref_val=0., point_estimate='mode')
