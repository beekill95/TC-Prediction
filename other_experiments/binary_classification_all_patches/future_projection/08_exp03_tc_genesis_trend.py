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
import numpy as np
import numpyro
from numpyro.infer import NUTS, MCMC
import os
import pandas as pd
from scipy.stats import poisson as poisson_rv
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
wfb_count_df['cluster'] = 'wbf'
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
mcmc.print_summary(exclude_deterministic=False)
# -

idata = az.from_numpyro(
    mcmc,
    coords=dict(rcp=['RCP45', 'RCP85'], cluster=count_df['cluster'].cat.categories),
    dims=dict(b2=['rcp'], b3=['cluster']))
az.plot_trace(idata)
plt.tight_layout()


# ### Plot Data Fit

# +
def plot_data_fit(idata, ax: plt.Axes, *, rcp: str, cluster: str, nb_lines: int = 20):
    posterior = idata['posterior']
    b0 = posterior['b0'].values.flatten()
    b1 = posterior['b1'].values.flatten()
    b2 = posterior['b2'].sel(rcp=rcp).values.flatten()
    b3 = posterior['b3'].sel(cluster=cluster).values.flatten()

    # Choose random lines.
    random_indices = np.random.choice(len(b0), size=nb_lines)
    year_range = np.linspace(2030, 2100, 100)
    years_with_dist = [2040, 2050, 2080, 2090, 2100]

    for idx in random_indices:
        mean = np.exp(b0[idx] + b1[idx] * year_range + b2[idx] + b3[idx])
        # Plot the mean lines.
        ax.plot(year_range, mean, color='blue', alpha=0.5)

        # Super-impose Poisson distributions.
        for yr in years_with_dist:
            rv = poisson_rv(np.exp(b0[idx] + b1[idx] * yr + b2[idx]))
            cnt = np.arange(rv.ppf(0.01), rv.ppf(0.99))
            pmf = rv.pmf(cnt)
            # Scale pmf.
            pmf = pmf * 9 / pmf.max()
            ax.plot(-pmf + yr, cnt, color='purple', alpha=0.5)

    for yr in years_with_dist:
        ax.vlines(yr, *ax.get_ylim(), linestyles='dashed', alpha=0.5)


fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
rcp45_dbscan_mask = (count_df['cluster'] == 'dbscan') & (count_df['rcp'] == 0)
ax.scatter(count_df[rcp45_dbscan_mask]['year'], count_df[rcp45_dbscan_mask]['genesis'], color='black')
plot_data_fit(idata, ax=ax, rcp='RCP45', cluster='dbscan')
# -

data_to_plot = [
    dict(cluster='dbscan', rcp='RCP45'),
    dict(cluster='dbscan', rcp='RCP85'),
    dict(cluster='wbf', rcp='RCP45'),
    dict(cluster='wbf', rcp='RCP85'),
]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), layout='constrained')
for ax, data, panel in zip(axes.flatten(), data_to_plot, ['a)', 'b)', 'c)', 'd)']):
    cluster, rcp = data['cluster'], data['rcp']
    mask = (count_df['cluster'] == cluster) & (count_df['rcp'] == (0 if rcp == 'RCP45' else 1))
    ax.scatter(count_df[mask]['year'], count_df[mask]['genesis'], color='black')
    plot_data_fit(idata, ax=ax, rcp=rcp, cluster=cluster)

    #ax.set_title(f'{cluster=} - {rcp=}')
    ax.set_xlabel('Year')
    ax.set_ylabel('TCGs')
    ax.set_title(panel, loc='left')

# ### Year

plt.figure(figsize=(4, 4))
az.plot_posterior(idata, var_names=['b1'], hdi_prob=0.95, ref_val=0., point_estimate='mode')
plt.title('')

# ### RCP Scenarios

# +
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), layout='constrained')
ax = axes[0]
az.plot_posterior(
    idata,
    var_names=['b2'],
    coords=dict(rcp='RCP45'),
    hdi_prob=0.95,
    ref_val=0.,
    point_estimate='mode',
    ax=ax)
ax.set_title('a)', loc='left')
ax.set_title('', loc='center')

ax = axes[1]
az.plot_posterior(
    idata,
    var_names=['b2'],
    coords=dict(rcp='RCP85'),
    hdi_prob=0.95,
    ref_val=0.,
    point_estimate='mode',
    ax=ax)
ax.set_title('b)', loc='left')
ax.set_title('', loc='center')
# -

# ### Clustering Method

# +
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), layout='constrained')
ax = axes[0]
az.plot_posterior(
    idata,
    var_names=['b3'],
    coords=dict(cluster='dbscan'),
    hdi_prob=0.95,
    ref_val=0.,
    point_estimate='mode',
    ax=ax)
ax.set_title('a)', loc='left')
ax.set_title('', loc='center')

ax = axes[1]
az.plot_posterior(
    idata,
    var_names=['b3'],
    coords=dict(cluster='wbf'),
    hdi_prob=0.95,
    ref_val=0.,
    point_estimate='mode',
    ax=ax)
ax.set_title('b)', loc='left')
ax.set_title('', loc='center')
# -

# ### Difference in genesis between the end and the mid-century

# +
import numpy as np # noqa


mid_century_years = np.arange(2030, 2051)
end_century_years = np.arange(2080, 2101)
print(mid_century_years)
# -

# Here, we will calculate the difference between number of storms between
# mid-century and end-century period,
# we do that by ignoring the coefficients associated with other factors
# (i.e. set these coefficients to 0.)

# +
# These will have the shape (nb_chains, nb_draws).
b0 = idata['posterior']['b0'].values
b1 = idata['posterior']['b1'].values

# And we want to calculate the mean for every year.
# The resulting array will be (nb_years, nb_chains, nb_draws)

# mean_nb_storms_mid_century = np.exp(b0[None, ...] + b1[None, ...] * mid_century_years[..., None, None])
# mean_nb_storms_end_century = np.exp(b0[None, ...] + b1[None, ...] * end_century_years[..., None, None])
# print(mean_nb_storms_mid_century.shape)

# Display the results of these years: 2030 vs 2080, 2040 vs 2090, and 2050 vs 2100.
titles = ['a)', 'b)', 'c)']
years_to_compare = [
    (2030, 2080),
    (2040, 2090),
    (2050, 2100),
]
fig, axes = plt.subplots(ncols=3, figsize=(12, 4), layout='constrained')
for title, (mid_year, end_year), ax in zip(titles, years_to_compare, axes.flatten()):
    # mid_idx, = np.where(mid_century_years == mid_year)
    # end_idx, = np.where(end_century_years == end_year)
    mean_end = np.exp(b0 + b1 * end_year)
    mean_mid = np.exp(b0 + b1 * mid_year)
    mean_diff = mean_end - mean_mid

    # mean_nb_storms_diff = mean_nb_storms_end_century[end_idx] - mean_nb_storms_mid_century[mid_idx]
    az.plot_posterior(
        mean_diff,
        ref_val=0.,
        hdi_prob=0.95,
        point_estimate='mode',
        ax=ax,
    )
    ax.set_title(title, loc='left')
    ax.set_title('', loc='center')
# -

# Instead of displaying the results like above,
# how about we show the difference between 2030 - 2100 in different RCP scenarios.

b2_rcp45 = idata['posterior']['b2'].sel(rcp='RCP45').values
b2_rcp85 = idata['posterior']['b2'].sel(rcp='RCP85').values
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), layout='constrained')
data = [
    ('a)', b2_rcp45, axes[0]),
    ('b)', b2_rcp85, axes[1]),
]
for title, b2, ax in data:
    mean_2030 = np.exp(b0 + b1 * 2030 + b2)
    mean_2100 = np.exp(b0 + b1 * 2100 + b2)
    mean_diff = mean_2100 - mean_2030
    az.plot_posterior(
        mean_diff,
        ref_val=0.,
        hdi_prob=0.95,
        point_estimate='mode',
        ax=ax,
    )
    ax.set_title(title, loc='left')
    ax.set_title('', loc='center')
