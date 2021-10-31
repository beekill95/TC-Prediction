# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import xarray as xr

data = xr.open_dataset("/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels/6h_700mb/fnl_20080501_00_00.nc")
data
