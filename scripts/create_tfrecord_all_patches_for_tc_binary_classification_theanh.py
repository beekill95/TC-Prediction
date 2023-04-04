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


import abc
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import pathlib
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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best-track',
        dest='best_track',
        required=True,
        help='Pattern of best track files.')
    parser.add_argument(
        '--indir',
        dest='indir',
        required=True,
        help='Path to .nc files extracted from The Anh data.')
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
        '--all-variables',
        dest='all_variables',
        action='store_true',
        help='Whether should we extract all variables or not. Default is False, which means only a subset is extracted.\
        This flag is ignored when netcdf files are extracted, which means netcdf files always contain all variables.'
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
        '--downscale',
        action='store_true',
        help='Whether to downscale the extracted subset to 1-degree resolution. Default to False.')
    parser.add_argument(
        '--from-date',
        dest='from_date',
        # default='20300101',
        type=parse_date_from_arg_string,
        help='Extract data from this date onward. Default is None, which means extract from the beginning.')
    parser.add_argument(
        '--till-date',
        dest='till_date',
        # default='20350101',
        type=parse_date_from_arg_string,
        help='Extract data to this date. Default is None, which means extract till the end.')
    parser.add_argument(
        '--outformat',
        default='tfrecords',
        choices=['tfrecords', 'netcdf'],
        help='Decide between `tfrecords` and `netcdf` output formats. Default is `tfrecords`.')
    parser.add_argument(
        '--out',
        required=True,
        help='Path to output file (tfrecords) or output directory (netcdf).')

    return parser.parse_args(args)


def parse_date_from_arg_string(d: str | None):
    return datetime.strptime(d, '%Y%m%d') if d is not None else None


def downscale_ds_to_1deg_resolution(ds: xr.Dataset) -> xr.Dataset:
    latmin, lonmin = tuple(round(ds[dim].values.min()) for dim in ['lat', 'lon'])
    latmax, lonmax = tuple(round(ds[dim].values.max()) for dim in ['lat', 'lon'])
    return ds.interp(
        lat=np.arange(latmin, latmax + 1),
        lon=np.arange(lonmin, lonmax + 1),
        method='linear',
        kwargs={"fill_value": "extrapolate"},)


def fill_missing_values(ds: xr.Dataset) -> xr.Dataset:
    mean_values = ds.mean(dim=['lat', 'lon'], skipna=True)
    return ds.fillna(mean_values)


def has_nan(ds: xr.Dataset, desc):
    for name, values in ds.items():
        isnan = np.isnan(values.values)
        if isnan.ndim == 3:
            isnan = np.any(isnan, axis=(1, 2))
        print(desc, name, isnan)


class OutputWriter(abc.ABC):
    @abc.abstractmethod
    def prepare_destination(self):
        pass

    @abc.abstractmethod
    def prepare_dataset(self, ds: xr.Dataset, location: tuple[float, float], is_genesis: bool, original_path: str, stormid: str):
        """
        This function will be called for each dataset patch that we extract.
        The result returned by this function will be appended into a list,
        which will be returned to a parent process.
        This function will be called by multiple parallel processes.
        """
        pass

    @abc.abstractmethod
    def write(self, tasks, nb_files: int, desc: str):
        """
        This function will be called in the parent process.
        The tasks is an iterator that stores the results of the `prepare_dataset()` function.
        In particular, each element in the iterator is a list containing the result `prepare_dataset` returned.
        """
        pass


class netcdf4Writer(OutputWriter):
    def __init__(self, outputdir: str) -> None:
        self._posdir = os.path.join(outputdir, 'pos')
        self._negdir = os.path.join(outputdir, 'neg')

    def prepare_destination(self):
        # Make sure that the folder does not exist.
        os.makedirs(self._posdir)
        os.makedirs(self._negdir)

    def prepare_dataset(self, ds: xr.Dataset, location: tuple[float, float], is_genesis: bool, original_path: str, stormid: str):
        """
        We don't have to prepare anything here, just save the data.
        """
        path = pathlib.Path(original_path)
        datetime_part = '_'.join(path.stem.split('_')[1:])
        lat, lon = location
        fn_prefix = f'{datetime_part}_{lat:.2f}_{lon:.2f}'
        save_path = (os.path.join(self._posdir, f'{fn_prefix}_{stormid}.nc')
                     if is_genesis
                     else os.path.join(self._negdir, f'{fn_prefix}.nc'))
        ds.to_netcdf(save_path)

    def write(self, tasks, nb_files: int, desc: str):
        """
        This function will be called in the parent process.
        """
        for _ in tqdm(tasks, total=nb_files, desc=desc):
            pass


class TfrecordWriter(OutputWriter):
    def __init__(self, outfile: str, extract_all_variables: bool, variables_order: list[str]) -> None:
        self._outfile = outfile
        self._variables_order = variables_order
        self._extract_all_variables = extract_all_variables

    def prepare_destination(self):
        # Make sure that the destination file does not exist.
        assert not os.path.isfile(self._outfile), f'Output file exists: {self._outfile}'

    def prepare_dataset(self, ds: xr.Dataset, location: tuple[float, float], is_genesis: bool, original_path: str, stormid: str):
        patch = (extract_subset(ds, SUBSET)
                 if not self._extract_all_variables
                 else extract_all_variables(ds, self._variables_order))
        example = self.to_example(patch, np.asarray(location), is_genesis, original_path, stormid)
        return example.SerializeToString()

    def to_example(self, value: np.ndarray, pos: np.ndarray, genesis: bool, path: str, stormid: str):
        feature = dict(
            data=numpy_feature(value),
            data_shape=int64_feature(value.shape),
            position=numpy_feature(pos),
            genesis=int64_feature([genesis]),
            filename=bytes_feature(str.encode(path)),
            stormid=bytes_feature(str.encode(stormid)),
        )
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write(self, tasks, nb_files: int, desc: str):
        with tf.io.TFRecordWriter(self._outfile) as writer:
            for results in tqdm(tasks, total=nb_files, desc=desc):
                for r in results:
                    writer.write(r)


@dataclass
class ProcessArgs:
    row: pd.Series
    domain_size: float
    stride: float
    downscale: bool
    writer: OutputWriter


def extract_dataset_samples(args: ProcessArgs) -> list[str]:
    row, domain_size, stride = args.row, args.domain_size, args.stride
    ds = xr.load_dataset(row['Path'], engine='netcdf4')
    ds = fill_missing_values(ds)
    if args.downscale:
        ds = downscale_ds_to_1deg_resolution(ds)
    lat, lon = ds['lat'].values, ds['lon'].values
    latmin, latmax = lat.min(), lat.max()
    lonmin, lonmax = lon.min(), lon.max()

    g_lat, g_lon = row['LAT'], row['LON']

    results = []
    writer = args.writer
    for lt in np.arange(latmin, latmax, stride):
        for ln in np.arange(lonmin, lonmax, stride):
            if ((lt + domain_size) > latmax) or ((ln + domain_size) > lonmax):
                continue

            if g_lat is None:
                genesis = False
            else:
                if not isinstance(g_lat, list):
                    g_lat = [g_lat]
                    g_lon = [g_lon]

                genesis = any(
                    (lt < glt < lt + domain_size) and (ln < gln < ln + domain_size)
                    for glt, gln in zip(g_lat, g_lon))

            patch = ds.sel(lat=slice(lt, lt + domain_size), lon=slice(ln, ln + domain_size))
            stormid = row['SID']
            stormid = '_'.join(stormid)if isinstance(stormid, list) else str(stormid)
            r = writer.prepare_dataset(patch, (lt, ln), genesis, str(row['Path']), stormid)
            results.append(r)

    return results


def extract_dataset_samples_parallel(
        genesis_df: pd.DataFrame, writer: OutputWriter, *,
        domain_size: float,
        stride: float,
        downscale: bool,
        processes: int,
        desc: str):
    with Pool(processes) as pool:
        tasks = pool.imap_unordered(
            extract_dataset_samples, 
            (ProcessArgs(
                row=r,
                domain_size=domain_size,
                stride=stride,
                downscale=downscale,
                writer=writer)
             for _, r in genesis_df.iterrows()))

        writer.prepare_destination()
        writer.write(tasks, nb_files=len(genesis_df), desc=desc)


def main(args=None):
    args = parse_args(args)
    
    outfile = args.out

    files = list_reanalysis_files(args.indir)
    genesis_df, _ = load_best_track_files_theanh(args.best_track)

    # Combine best track with data that we have.
    # In this step, all negative samples
    # (observations without TC genesis) are removed.
    files['OriginalDate'] = files['Date'].copy()
    files['Date'] = files['Date'].apply(
        lambda date: date + timedelta(hours=args.leadtime))
    genesis_df = files.merge(genesis_df, how='inner', on='Date')
    genesis_df = genesis_df.groupby('Path').agg({
        'OriginalDate': 'first',
        'LAT': lambda x: x.iloc[0] if len(x) == 1 else list(x),
        'LON': lambda x: x.iloc[0] if len(x) == 1 else list(x),
        'SID': lambda x: x.iloc[0] if len(x) == 1 else list(x), 
        'Date': lambda x: x.iloc[0] if len(x) == 1 else list(x), 
    })
    genesis_df['Path'] = genesis_df.index

    dates = genesis_df['OriginalDate']
    if args.from_date is not None:
        genesis_df = genesis_df[dates >= args.from_date]
    if args.till_date is not None:
        genesis_df = genesis_df[dates < args.till_date]

    # Create an appropriate writer.
    variables_order = list(VARIABLES_ORDER)
    variables_order.remove('capesfc')
    writer = (netcdf4Writer(outfile)
              if args.outformat == 'netcdf'
              else TfrecordWriter(outfile, args.all_variables, variables_order))

    # Extract datasets.
    assert not os.path.isfile(outfile), f'Output file: {outfile=} exists!'
    extract_dataset_samples_parallel(
        genesis_df,
        writer,
        domain_size=args.domain_size,
        stride=args.stride,
        downscale=args.downscale,
        processes=args.processes,
        desc=f'Extracting from {args.from_date} to {args.till_date}')


if __name__ == '__main__':
    main()
