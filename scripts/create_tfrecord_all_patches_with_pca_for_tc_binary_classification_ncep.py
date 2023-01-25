#!/usr/bin/env python3

"""
This script will extract all patches in a .nc file from NCEP/FNL data
to create binary genesis classification dataset in tfrecords format
and also perform PCA on the data.

PCA is first performed on train dataset using sklearn Incremental PCA,
then applied to other datasets.
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
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm
import xarray as xr


VARIABLES_ORDER = [
    'absvprs',
    'capesfc',
    'hgtprs',
    'pressfc',
    'rhprs',
    'tmpprs',
    'tmpsfc',
    'ugrdprs',
    'vgrdprs',
    'vvelprs',
]
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
        '--nb-pca',
        dest='nb_pca',
        type=int,
        help='Number of principal components.',
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


def extract_values(ds: xr.Dataset, order: list[str]):
    values = []

    for varname in order:
        var = ds[varname].values
        if var.ndim == 2:
            var = var[None, ...]

        values.append(var)

    values = np.concatenate(values, axis=0)
    values = np.moveaxis(values, 0, 2)
    return values


def to_example(value: np.ndarray, pos: np.ndarray, genesis: bool, path: str):
    feature = dict(
        data=numpy_feature(value, dtype=np.float32),
        data_shape=int64_feature(value.shape),
        position=numpy_feature(pos, dtype=np.float32),
        genesis=int64_feature([genesis]),
        filename=bytes_feature(str.encode(path)),
    )
    return tf.train.Example(features=tf.train.Features(feature=feature))


ProcessArgs = namedtuple('ProcessArgs', ['row', 'domain_size', 'stride', 'scaler', 'pca'])
def extract_dataset_samples(args: ProcessArgs) -> list[str]:
    row, domain_size, stride, scaler, pca = args
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
            patch = extract_values(patch, VARIABLES_ORDER)

            # Reshape patch so that we can perform PCA.
            patch_original_shape = patch.shape
            patch = patch.reshape((-1, patch_original_shape[-1]))
            patch = scaler.transform(patch)
            patch = pca.transform(patch)
            patch = patch.reshape(tuple(patch_original_shape[:2]) + (-1,))

            patch_example = to_example(
                patch, np.asarray([lt, ln]), genesis, row['Path'])
            results.append(patch_example.SerializeToString())

    return results


def extract_dataset_samples_parallel(
        genesis_df: pd.DataFrame, outputfile: str, *,
        domain_size: float, stride: float, processes: int, desc: str,
        scaler: StandardScaler, pca: IncrementalPCA):
    with Pool(processes) as pool:
        tasks = pool.imap_unordered(
            extract_dataset_samples, 
            (ProcessArgs(r, domain_size, stride, scaler, pca)
             for _, r in genesis_df.iterrows()))

        with tf.io.TFRecordWriter(outputfile) as writer:
            for results in tqdm(tasks, total=len(genesis_df), desc=desc):
                for r in results:
                    writer.write(r)


def load_path(path: str):
    ds = xr.load_dataset(path, engine='netcdf4')
    values = extract_values(ds, VARIABLES_ORDER)
    nb_variables = values.shape[-1]
    return values.reshape(-1, nb_variables)


def perform_pca(genesis_df: pd.DataFrame, n_components: int, scaler: StandardScaler):
    files = genesis_df['Path'].unique()
    pca = IncrementalPCA(n_components)

    with ThreadPool(4) as pool:
        tasks = pool.imap_unordered(load_path, files)

        for values in tqdm(tasks, total=len(files), desc='PCA'):
            values = scaler.transform(values)
            pca.partial_fit(values)

    return pca


def standard_scaler(genesis_df: pd.DataFrame):
    files = genesis_df['Path'].unique()
    scaler = StandardScaler()

    with ThreadPool(4) as pool:
        tasks = pool.imap_unordered(load_path, files)

        for values in tqdm(tasks, total=len(files), desc='Standard Scaler'):
            scaler.partial_fit(values)

    return scaler


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

    scaler = standard_scaler(train_genesis_df)
    pca = perform_pca(train_genesis_df, args.nb_pca, scaler)
    print(f'{args.nb_pca} chosen principal components explain {pca.explained_variance_ratio_.sum()}')
    print(pca.explained_variance_ratio_)

    # Create output directories.
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)

    tasks = [
        ('Train', train_genesis_df),
        ('Val', val_genesis_df),
        ('Test', test_genesis_df),
    ]

    # # Extract datasets.
    for desc, df in tasks:
        fn, ext = os.path.splitext(os.path.basename(outfile))
        path = os.path.join(outdir, f'{fn}_{desc}{ext}')
        assert not os.path.isfile(path), f'Output file: {outfile=} exists!'

        df.to_csv(f'tfrecords_{desc}.csv')

        extract_dataset_samples_parallel(
            df, path,
            domain_size=args.domain_size,
            stride=args.stride,
            processes=args.processes,
            desc=desc,
            pca=pca,
            scaler=scaler)


if __name__ == '__main__':
    main()
