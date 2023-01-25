#!/usr/bin/env python3

"""
This script will extract all patches in a .nc file from NCEP/FNL data
to create binary genesis classification dataset in tfrecords format.
"""
try:
    from .tc_binary_classification_helpers import *
except ImportError:
    from tc_binary_classification_helpers import *

try:
    from .tfrecords_utils import *
except ImportError:
    from tfrecords_utils import *


import argparse
from collections import OrderedDict, namedtuple
from datetime import datetime, timedelta
from functools import reduce
import glob
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import xarray as xr


SUBSET = OrderedDict(
    absvprs=(900, 750),
    rhprs=(750,),
    tmpprs=(900, 500),
    hgtprs=(500,),
    vvelprs=(500,),
    ugrdprs=(800, 200),
    vgrdprs=(800, 200),
    tmpsfc=True,
    pressfc=True,
)
TRAIN_DATE_END = datetime(2016, 1, 1)
VAL_DATE_END = datetime(2018, 1, 1)


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
        help='Path to NCEP/FNL .nc files.')
    parser.add_argument(
        '--basin',
        nargs='+',
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
        '--processes',
        default=8,
        type=int,
        help='Number of parallel processes. Default to 8.'
    )
    parser.add_argument(
        '--domain-size',
        dest='domain_size',
        default=30,
        type=float,
        help='Size (in degrees) of the extracted domain. Default is 30deg.')
    parser.add_argument(
        '--stride',
        default=5,
        type=float,
        help='Stride (in degrees). Default is 5deg.')
    parser.add_argument(
        '--outfile',
        required=True,
        help='Path to output file.')

    return parser.parse_args(args)


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '*.nc'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths,
    })


def extract_subset(ds: xr.Dataset, subset: OrderedDict) -> np.ndarray:
    tensors = []
    for key, lev in subset.items():
        values = None
        if isinstance(lev, bool):
            if lev:
                values = ds[key].values
        else:
            try:
                values = ds[key].sel(lev=list(lev)).values
            except Exception as e:
                print(key, lev, ds[key]['lev'])
                raise e

        if values is not None:
            if values.ndim == 2:
                values = values[None, ...]

            tensors.append(values)

    tensors = np.concatenate(tensors, axis=0)
    tensors = np.moveaxis(tensors, 0, -1)
    return tensors

def to_example(value: np.ndarray, pos: np.ndarray, genesis: bool, path: str):
    feature = dict(
        data=numpy_feature(value),
        data_shape=int64_feature(value.shape),
        position=numpy_feature(pos),
        genesis=int64_feature([genesis]),
        filename=bytes_feature(str.encode(path)),
    )
    return tf.train.Example(features=tf.train.Features(feature=feature))

ProcessArgs = namedtuple('ProcessArgs', ['row', 'domain_size', 'stride'])
def extract_dataset_samples(args: ProcessArgs) -> list[str]:
    row, domain_size, stride = args
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    lat, lon = ds['lat'].values, ds['lon'].values
    latmin, latmax = lat.min(), lat.max()
    lonmin, lonmax = lon.min(), lon.max()

    g_lat, g_lon = row['LAT'], row['LON']

    results = []
    for lt in np.arange(latmin, latmax, stride):
        for ln in np.arange(lonmin, lonmax, stride):
            if ((lt + domain_size) > latmax) or ((ln + domain_size) > lonmax):
                continue

            if g_lat is None:
                genesis = False
            else:
                genesis = (lt < g_lat < lt + domain_size) and (ln < g_lon < ln + domain_size)

            patch = ds.sel(lat=slice(lt, lt + domain_size), lon=slice(ln, ln + domain_size))
            patch = extract_subset(patch, SUBSET)
            patch_example = to_example(
                patch, np.asarray([lt, ln]), genesis, row['Path'])
            results.append(patch_example.SerializeToString())

    return results


def extract_dataset_samples_parallel(
        genesis_df: pd.DataFrame, outputfile: str, *,
        domain_size: float, stride: float, processes: int, desc: str):
    with Pool(processes) as pool:
        tasks = pool.imap_unordered(
            extract_dataset_samples, 
            (ProcessArgs(r, domain_size, stride) for _, r in genesis_df.iterrows()))

        with tf.io.TFRecordWriter(outputfile) as writer:
            for results in tqdm(tasks, total=len(genesis_df), desc=desc):
                for r in results:
                    writer.write(r)


def main(args=None):
    args = parse_args(args)
    
    outfile = args.outfile

    files = list_reanalysis_files(args.ncep_fnl)
    genesis_df, _ = load_best_track(args.best_track)

    # Filter out basins.
    # storms_df = storms_df[storms_df['BASIN'].isin(args.basin)]
    genesis_df = genesis_df[genesis_df['BASIN'].isin(args.basin)]

    # Remove storms that are outside the domain of interest.
    ds = xr.load_dataset(files['Path'].iloc[0], engine='netcdf4')
    lat, lon = ds['lat'].values, ds['lon'].values
    latmin, latmax = lat.min(), lat.max()
    lonmin, lonmax = lon.min(), lon.max()
    genesis_df = genesis_df[
        (latmin <= genesis_df['LAT']) & (genesis_df['LAT'] <= latmax)
        & (lonmin <= genesis_df['LON']) & (genesis_df['LON'] <= lonmax)]

    # Combine best track with data that we have.
    # In this step, all negative samples
    # (observations without TC genesis) are removed.
    files['OriginalDate'] = files['Date'].copy()
    files['Date'] = files['Date'].apply(
        lambda date: date + timedelta(hours=args.leadtime))
    genesis_df = files.merge(genesis_df, how='inner', on='Date')


    # Split into train, validation, and test datasets.
    dates = genesis_df['OriginalDate']
    train_genesis_df = genesis_df[dates < TRAIN_DATE_END]
    val_genesis_df = genesis_df[(dates >= TRAIN_DATE_END) & (dates < VAL_DATE_END)]
    test_genesis_df = genesis_df[dates > VAL_DATE_END]

    # Create output directories.
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)

    tasks = [
        ('Train', train_genesis_df),
        ('Val', val_genesis_df),
        ('Test', test_genesis_df),
    ]

    # Extract datasets.
    for desc, df in tasks:
        fn, ext = os.path.splitext(os.path.basename(outfile))
        path = os.path.join(outdir, f'{fn}_{desc}{ext}')
        assert not os.path.isfile(path), f'Output file: {outfile=} exists!'

        df.to_csv(f'tfrecords_{desc}.csv')

        extract_dataset_samples_parallel(
            df, path,
            domain_size=args.domain_size, stride=args.stride,
            processes=args.processes, desc=desc)


if __name__ == '__main__':
    main()
