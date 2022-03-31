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

import matplotlib.pyplot as plt
import pandas as pd
from tc_formation.plots import observations as plt_obs

# # Daily Genesis Potential Test 01
#
# What is this test about?
#
# TODO:

label_path = 'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v4.csv'
label_df = pd.read_csv(label_path)
label_with_tc_df = label_df[label_df['TC']]

# ## Genesis Potential Index
#
# Based on the formula shown in [Gray](https://mountainscholar.org/bitstream/handle/10217/247/0234_Bluebook.pdf;sequence=1)
#
# (Genesis Potential) ∝ (Vorticity Parameter) (Corriolis Parameter) (Vertical Shear Parameter) (Ocean Thermal Energy) (Moist Stability Parameter) (Humidity Parameter)
#
# These parameters are further grouped into 2 categories:
#
# * Dynamic Potential: Vorticity parameter, Corriolis parameter, and Vertical Shear parameter.
# * Thermal Potential: Ocean Thermal energy, Moist Stability parameter, and Humidity parameter.
#
# Details of these parameters:
#
# * Vorticity parameter: $(\zeta_r + 5)$ at 950mb where $\zeta_r$ is in $10^{-6} \text{s}^{-1}$
# * Corriolis parameter: $2\Omega sin\phi$ where $\Omega$ is rotation rate of the Earth,
# and $\phi$ is the latitude.
# * Vertical Shear parameter: $1 / (S_z + 3)$ where $S_z = |\partial V / \partial p|$
# where $S_z$ is in m/s per 750mb.
# * Ocean Thermal energy: $\int_{60m}^{sfc} \rho_w c_w (T - 26)$
# where $\rho_w$ and $c_w$ are density and specific heat capacity of water respectively.
# E is in $10^3 cal/m^3$.
# * Moist Stability parameter: $\partial \theta_e / \partial p + 5$ is in K per 500mb.
# * Relative Humidity parameter: $\frac{\overline{RH} - 40}{30}$
# where $\overline{RH}$ is mean relative humidity between 700mb and 500mb.
# Parameter is 0 for $\overline{RH} < 40$ and 1 for $\overline{RH} \ge 70$.

# +
import tc_formation.genesis_potential.genesis_potential_index as gpi
