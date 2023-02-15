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

import pandas as pd

# Count TCG on RCP45 Predictions

pred_df = pd.read_csv('./06_exp01_future_projection_RCP85.csv')
pred_df.head()

# Just to make sure that there are positive predictions.

pos_pred_df = pred_df[pred_df['pred'] > 0.1]
pos_pred_df.head()
