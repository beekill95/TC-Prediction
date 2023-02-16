#!/usr/bin/env python3

"""
This script will convert NCEP/FNL data file into tfrecord file.
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
from collections import namedtuple
from datetime import datetime, timedelta
import glob
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import xarray as xr


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best-track',
        dest='best_track',
        required=True,
        help='Pattern of the best track files.')
    parser.add_argument(
        '--indir',
        required=True,
        help='Path to .nc files.')
    parser.add_argument(
        '--from-date',
        dest='from_date',
        default='20300101',
        type=lambda d: datetime.strptime(d, '%Y%m%d'),
        help='Extract data from this date onward.')
    parser.add_argument(
        '--till-date',
        dest='till_date',
        default='20350101',
        type=lambda d: datetime.strptime(d, '%Y%m%d'),
        help='Extract data to this date.')
    parser.add_argument(
        '--processes',
        default=8,
        type=int,
        help='Number of parallel processes. Default to 8.'
    )
    parser.add_argument(
        '--leadtime',
        default=6,
        type=int,
        help='Leadtime. Default to 6h.',
    )
    parser.add_argument(
        '--include-non-genesis',
        dest='include_non_genesis',
        action='store_true',
        help='Whether should we include non genesis files. Default to False.')
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


def to_example(value: np.ndarray, *, genesis_locations: np.ndarray, genesis_date: datetime, file_date: datetime, path: str):
    feature = dict(
        data=numpy_feature(value, dtype=np.float32),
        data_shape=int64_feature(value.shape),
        genesis_locations=numpy_feature(genesis_locations, dtype=np.float32),
        genesis_locations_shape=int64_feature(genesis_locations.shape),
        filename=bytes_feature(str.encode(path)),
        genesis_date=date_feature(genesis_date),
        file_date=date_feature(file_date),
    )
    return tf.train.Example(features=tf.train.Features(feature=feature))


def fill_missing_values(ds: xr.Dataset) -> xr.Dataset:
    mean_values = ds.mean(dim=['lat', 'lon'], skipna=True)
    return ds.fillna(mean_values)


def downscale_ds_to_1deg_resolution(ds: xr.Dataset) -> xr.Dataset:
    latmin, lonmin = tuple(round(ds[dim].values.min()) for dim in ['lat', 'lon'])
    latmax, lonmax = tuple(round(ds[dim].values.max())  for dim in ['lat', 'lon'])
    return ds.interp(
        lat=np.arange(latmin, latmax + 1),
        lon=np.arange(lonmin, lonmax + 1),
        method='linear',
        kwargs={"fill_value": "extrapolate"},)


ProcessArgs = namedtuple('ProcessArgs', ['row'])
def convert_nc_file_to_tfrecord(args: ProcessArgs):
    row = args.row
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    ds = fill_missing_values(ds)
    ds = downscale_ds_to_1deg_resolution(ds)
    lat, lon = ds['lat'].values, ds['lon'].values
    latmin, lonmin = lat.min(), lon.min()
    variables = list(VARIABLES_ORDER)
    variables.remove('capesfc')
    values = extract_all_variables(ds, variables)

    genesis_locations = [(lat, lon) for lat, lon in zip(row['LAT'], row['LON'])]
    example = to_example(
        values,
        genesis_locations=np.asarray(genesis_locations) - np.asarray([latmin, lonmin]),
        genesis_date=row['Date_genesis'],
        file_date=row['Date_file'],
        path=row['Path'])
    return example.SerializeToString()


def main(args=None):
    args = parse_args(args)
    outfile = args.outfile

    files_df = list_reanalysis_files(args.indir)
    genesis_df, _ = load_best_track_files_theanh(args.best_track)

    # Remove storms that are outside the domain of interest.
    # ds = xr.load_dataset(files_df['Path'].iloc[0], engine='netcdf4')
    # lat, lon = ds['lat'].values, ds['lon'].values
    # latmin, latmax = lat.min(), lat.max()
    # lonmin, lonmax = lon.min(), lon.max()
    # genesis_df = genesis_df[
    #     (latmin <= genesis_df['LAT']) & (genesis_df['LAT'] <= latmax)
    #     & (lonmin <= genesis_df['LON']) & (genesis_df['LON'] <= lonmax)]

    # For The Anh's dataset,
    # we only care about observations at 0h.
    hours = files_df['Date'].apply(lambda d: d.hour)
    files_df = files_df[hours == 0]

    # Merge the files and genesis_df.
    files_df['Date_leadtime'] = files_df['Date'].apply(
        lambda date: date + timedelta(hours=args.leadtime))
    genesis_df = files_df.merge(
        genesis_df,
        how='outer' if args.include_non_genesis else 'inner',
        left_on='Date_leadtime',
        right_on='Date',
        suffixes=('_file', '_genesis'))
    genesis_df = genesis_df.drop(columns=['Date_leadtime'])
    genesis_df = genesis_df.groupby('Path').agg({
        'Date_file': 'first',
        'Date_genesis': 'first', 
        'LAT': list,
        'LON': list,
        # 'BASIN': lambda x: x.iloc[0] if len(x) == 1 else list(x), 
        'SID': lambda x: x.iloc[0] if len(x) == 1 else list(x), 
    })
    genesis_df['Path'] = genesis_df.index

    # Filter files based on given date arguments.
    genesis_df = genesis_df[
            (genesis_df['Date_file'] >= args.from_date)
            & (genesis_df['Date_file'] < args.till_date)]

    # Process these files in parallel.
    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            convert_nc_file_to_tfrecord,
            [ProcessArgs(row) for _, row in genesis_df.iterrows()])

        assert not os.path.isfile(outfile), f'Output file exists! {outfile=}'
        with tf.io.TFRecordWriter(outfile) as writer:
            for result in tqdm(tasks, total=len(genesis_df), desc='Extracting'):
                writer.write(result)

if __name__ == '__main__':
    main()
