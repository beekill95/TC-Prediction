#!/bin/env python3

"""
This script will create binary classification dataset for tropical cyclogenesis.
"""

from __future__ import annotations

try:
    from .tc_binary_classification_helpers import *
except ImportError:
    from tc_binary_classification_helpers import *

import argparse
import glob
from multiprocessing import Pool
import os
import pandas as pd
from tqdm import tqdm
import xarray as xr


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best-track',
        dest='best_track',
        required=True,
        help='Path to ibtracs best track.')
    parser.add_argument(
        '--theanh-baseline',
        dest='theanh_baseline',
        required=True,
        help='Path to The Anh\'s baseline output.')
    parser.add_argument(
        '--domain-size',
        dest='domain_size',
        default=30,
        type=float,
        help='Size (in degrees) of the extracted domain. Default is 30deg.')
    parser.add_argument(
        '--distances',
        nargs='+',
        default=50, # 5000km
        type=float,
        help='Distance (in degrees) of the negative domain\'s center to positive domain\'s center. Default is 50deg.')
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output directory.')

    return parser.parse_args(args)


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '*.nc'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths
    })

class TheAnhPositiveNegativePatchesExtract(PositiveAndNegativePatchesExtractor):
    def load_dataset(self, path: str) -> xr.Dataset:
        return xr.load_dataset(path)


def main(args=None):
    args = parse_args(args)
    files = list_reanalysis_files(args.theanh_baseline)
    best_track = load_best_track(args.best_track)

    # Combine best track with data that we have.
    # In this step, all negative samples (observations without TC) are removed.
    best_track = files.merge(best_track, how='inner', on='Date')

    # Create output directories.
    os.makedirs(pos_output_dir(args.output), exist_ok=True)
    os.makedirs(neg_output_dir(args.output), exist_ok=True)

    # Now, loop over all files and extract the patches.
    with Pool() as pool:
        tasks = pool.imap_unordered(
            TheAnhPositiveNegativePatchesExtract(raise_cannot_find_negative_patch=False),
            (ExtractPosNegFnArgs(row, args.domain_size, args.distances, args.output)
             for _, row in best_track.iterrows()))

        # Loop through tasks so they get executed.
        # Also, show the progress along the way.
        for _ in tqdm(tasks, total=len(best_track)):
            pass

if __name__ == '__main__':
    main()
