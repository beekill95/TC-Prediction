#!/bin/env python3

"""
This script will create binary classification dataset for tropical cyclogenesis.
The reanalysis data for this only the NCEP/FNL dataset.
"""

try:
    from .tc_binary_classification_helpers import *
except ImportError:
    from tc_binary_classification_helpers import *

import argparse
from datetime import timedelta
from functools import reduce
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
        '--ncep-fnl',
        dest='ncep_fnl',
        required=True,
        help='Path to NCEP/FNL reanalysis.')
    parser.add_argument(
        '--basin',
        required=True,
        choices=['WP', 'EP', 'NA'],
        help='Basin to extract the storm. Accepted basins are: WP, EP, and AL.')
    parser.add_argument(
        '--leadtime',
        default=0,
        type=int,
        help='Lead time (in hours). Default is 0h.',
    )
    parser.add_argument(
        '--domain-size',
        dest='domain_size',
        default=30,
        type=float,
        help='Size (in degrees) of the extracted domain. Default is 30deg.')
    parser.add_argument(
        '--distance',
        default=50,
        type=float,
        help='Distance (in degrees) of the negative domain\'s center to positive domain\'s center. Default is 50deg.')
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output directory.')

    return parser.parse_args(args)


class NCEP_FNL_PositiveNegativePatchesExtractor(PositiveAndNegativePatchesExtractor):
    def load_ds(self, path, filter_by_keys: dict) -> xr.Dataset:
        return xr.load_dataset(
            path,
            engine='cfgrib',
            backend_kwargs=dict(
                errors='raise',
                filter_by_keys=filter_by_keys,
            ),
        )

    def load_dataset(self, path: str) -> xr.Dataset:
        def load_common_ds() -> xr.Dataset:
            variables = ['u', 'v', 'w', 't', 'r', 'gh', 'absv']
            datasets = [
                self.load_ds(
                    path, filter_by_keys=dict(typeOfLevel='isobaricInhPa', shortName=v))
                for v in variables]

            ds = reduce(lambda acc, cur: acc.merge(cur), datasets[1:], datasets[0])
            assert len(ds['isobaricInhPa'].values) > 0, f'Empty pressure levels for {path}'
            ds = ds.where((ds['isobaricInhPa'] >= 200) & (ds['isobaricInhPa'] <= 1e3), drop=True)
            return ds

        common_ds = load_common_ds()

        surface_ds = self.load_ds(
            path, filter_by_keys=dict(typeOfLevel='surface'))
        surface_ds = surface_ds.rename_vars(dict(t='tsfc'))

        # Merge datasets.
        merged_ds = common_ds.merge(surface_ds)

        rename_vars = dict(
            u='ugrdprs',
            v='vgrdprs',
            w='vvelprs',
            absv='absvprs',
            t='tmpprs',
            tsfc='tmpsfc',
            sp='pressfc',
            gh='hgtprs',
            cape='capesfc',
            r='rhprs',
            lsm='landmask',
        )

        # Rename variables to what we usually do.
        merged_ds = merged_ds.rename_vars(rename_vars)

        # Only retain what we care.
        remove_vars = [var.name
                       for var in merged_ds.data_vars.values()
                       if var.name not in rename_vars.values()]
        merged_ds = merged_ds.drop_vars(remove_vars)

        # Rename coordinates.
        merged_ds = merged_ds.rename(
            dict(latitude='lat', longitude='lon', isobaricInhPa='lev'))

        return merged_ds


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '**', '*.grib2'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths
    })


def main(args=None):
    args = parse_args(args)
    files = list_reanalysis_files(args.ncep_fnl)
    genesis_df, storms_df = load_best_track(args.best_track)

    # Filter out basins.
    storms_df = storms_df[storms_df['BASIN'] == args.basin]
    genesis_df = genesis_df[genesis_df['BASIN'] == args.basin]
    
    # Combine best track with data that we have.
    # In this step, all negative samples (observations without TC) are removed.
    files['OriginalDate'] = files['Date'].copy()
    files['Date'] = files['Date'].apply(
        lambda date: date + timedelta(hours=args.leadtime))
    genesis_df = files.merge(genesis_df, how='inner', on='Date')

    # DEBUG:
    # best_track = best_track[
    #     best_track['Path'].apply(lambda path: '20170826_18_00' in path)
    # ]

    # Create output directories.
    os.makedirs(pos_output_dir(args.output))
    os.makedirs(neg_output_dir(args.output))

    # Now, loop over all files and extract the patches.
    with Pool() as pool:
        tasks = pool.imap_unordered(
            NCEP_FNL_PositiveNegativePatchesExtractor(detailed_best_track=storms_df),
            (ExtractPosNegFnArgs(row, args.domain_size, [args.distance], args.output)
             for _, row in genesis_df.iterrows()))

        # Loop through tasks so they get executed.
        # Also, show the progress along the way.
        for _ in tqdm(tasks, total=len(genesis_df)):
            pass

if __name__ == '__main__':
    main()
