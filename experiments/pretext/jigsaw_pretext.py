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

# +
# %cd ../..
# %load_ext autoreload
# %autoreload 2

# -

# # Jigsaw Pretext
#
# In this experiment,
# I will try experimenting with pretext tasks.
# Specifically, I will use [jigsaw](https://arxiv.org/abs/1603.09246)
# as pretext:
# 
# * Divide the observations into 20x20 degrees tiles.
# * Permutate them
# * Use a convolutional model to predict the which permutation were applied.
