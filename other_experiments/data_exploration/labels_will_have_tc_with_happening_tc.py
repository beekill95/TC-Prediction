# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.12
# ---

# %cd ../..

import pandas as pd

# # Analyze Labels Will Have TC with Happening TC

data = pd.read_csv('data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_100_260/12h/tc_ibtracs_12h_WP_EP_v3.csv')

# First, we will count how many "will happen TC"

will_happen_tc_cnt = data['TC'].values.sum()
will_happen_tc_cnt

# Then, will count how many "occurring TC"

other_happening_tc_cnt = data['Is Other TC Happening'].values.sum()
other_happening_tc_cnt

# Of all these "will happen TC" and "other occuring TC",
# how many of them are overlapped?

will_happen_overlap_other_cnt = (data['TC'].values & data['Is Other TC Happening'].values).sum()
will_happen_overlap_other_cnt

# $\Rightarrow$ So most of the "will happen" TCs are overlapped with the "other happening" TCs.
# If the model confuses between "other happening" TCs with "will happen" TCs,
# then the accuracy would be?

will_happen_overlap_other_cnt / will_happen_tc_cnt

# ---
#
# How many data do we have?

total_data_cnt = len(data)
total_data_cnt

# So, most of the time, there are days where TCs occuring somewhere in the ocean,
# and therefore, not having TCs days are actually a rare case.

total_data_cnt - other_happening_tc_cnt

other_happening_tc_cnt / total_data_cnt
