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
from tc_formation.tcg_analysis.clustering import DBScanClustering


numpyro.set_host_device_count(4)
# -

# Set font size for all matplotlib figures.
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=20)


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

clustering = DBScanClustering(0.6)
rcp45_count_df = clustering.count_genesis(rcp45_df)
rcp45_count_df.head()

rcp85_count_df = clustering.count_genesis(rcp85_df)
rcp85_count_df.head()

# Now, we can display the genesis trend.

# +
def display_genesis_trend(count_df: pd.DataFrame, ax: plt.Axes):
    mid_century_df = count_df[count_df['year'] <= 2050]
    end_century_df = count_df[count_df['year'] > 2050]

    nb_years = len(mid_century_df)
    ax.plot(range(nb_years), mid_century_df['genesis'], label='2030-2050', lw=4.)
    ax.plot(range(nb_years), end_century_df['genesis'], label='2080-2100', c='black', lw=4.)
    ax.legend()
    ax.set_xticks(range(nb_years))
    ax.set_xticklabels(f'{2030 + i}\n{2080 + i}' for i in range(nb_years))
    ax.set_xlabel('Year')
    ax.set_ylabel('Genesis Frequency')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)


_, axes = plt.subplots(figsize=(18, 12), layout='constrained', nrows=2, sharey=True)
ax = axes[0]
display_genesis_trend(rcp45_count_df, ax=ax)
ax.set_title('a)', loc='left', fontsize='xx-large')
ax = axes[1]
display_genesis_trend(rcp85_count_df, ax=ax)
ax.set_title('b)', loc='left', fontsize='xx-large')
# -

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

# We can plot the highest density region,
# and equivalent region.

az.plot_posterior(idata, var_names=['b1'], hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP45'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

az.plot_posterior(idata, var_names=['b2'], coords=dict(rcp='RCP85'), hdi_prob=0.95, ref_val=0., point_estimate='mode')

# We're going to plot the lines and distributions.

# +
import numpy as np # noqa
from scipy.stats import poisson # noqa


# First, plot the data points.
fig, ax = plt.subplots()
ax.scatter(rcp45_count_df['year'], rcp45_count_df['genesis'], color='black')

# Then, randomly choose 20 lines to plot.
posterior = idata['posterior']
b0 = posterior['b0'].values.flatten()
b1 = posterior['b1'].values.flatten()
b2 = posterior['b2'].sel(rcp='RCP45').values.flatten()

n_random_lines = 20
random_indices = np.random.choice(len(b0), size=n_random_lines)
year_range = np.linspace(2030, 2100, 100)
years_with_dist = [2040, 2050, 2080, 2100]
for idx in random_indices:
    mean = np.exp(b0[idx] + b1[idx] * year_range + b2[idx])
    # Plot the mean lines.
    ax.plot(year_range, mean, color='blue')

    # Super-impose Poisson distributions.
    for yr in years_with_dist:
        rv = poisson(np.exp(b0[idx] + b1[idx] * yr + b2[idx]))
        cnt = np.arange(rv.ppf(0.05), rv.ppf(0.95))
        pmf = rv.pmf(cnt)
        # Scale pmf.
        pmf = pmf * 9 / pmf.max()
        ax.plot(-pmf + yr, cnt, color='purple')

print(ax.get_ylim())
for yr in years_with_dist:
    ax.vlines(yr, *ax.get_ylim(), linestyles='dashed')

fig.tight_layout()
