#!/bin/env python3

"""
This is the python version of the `extract_environmental_features_without_labels.py`
It will produce more consistent output to the original dataset.
This should be used for future instead of the old one.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
from datetime import datetime
from functools import reduce
import glob
from multiprocessing import Pool
import os
import pandas as pd
import tc_binary_classification_helpers as helpers
from tqdm import tqdm
import xarray as xr


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ncep_fnl',
        help='Path to directory containing NCEP/FNL dataset')
    parser.add_argument(
        'outputdir',
        help='Path to output directory.')
    parser.add_argument(
        '--lat',
        required=True,
        nargs=2,
        help='Range of latitudes of the to be extracted domain. (min lat, max lat)')
    parser.add_argument(
        '--lon',
        required=True,
        nargs=2,
        help='Range of longitudes of the to be extracted domain. (min lon, max lon)')

    return parser.parse_args(args)


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '**', '*.grib2'))
    files = ((helpers.parse_date_from_nc_filename(f), f)
             for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths
    })

def load_grib2(path, filter_by_keys: dict) -> xr.Dataset:
    return xr.load_dataset(
        path,
        engine='cfgrib',
        backend_kwargs=dict(
            errors='raise',
            filter_by_keys=filter_by_keys,
        ),
    )

def load_dataset(path: str) -> xr.Dataset:
    def load_common_ds() -> xr.Dataset:
        variables = ['u', 'v', 'w', 't', 'r', 'gh', 'absv']
        datasets = [
            load_grib2(
                path, filter_by_keys=dict(typeOfLevel='isobaricInhPa', shortName=v))
            for v in variables]

        ds = reduce(lambda acc, cur: acc.merge(cur), datasets[1:], datasets[0])
        return ds.sel(isobaricInhPa=slice(200, 1e3))

    common_ds = load_common_ds()

    surface_ds = load_grib2(
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


ExtractDomainArgs = namedtuple(
    'ExtractDomainArgs',
    ['file', 'outputdir', 'latmin', 'latmax', 'lonmin', 'lonmax'])


def extract_domain(args: ExtractDomainArgs):
    file = args.file

    ds = load_dataset(file['Path'])
    domain_position = helpers.PatchPosition(
        lat_min = args.latmin,
        lat_max = args.latmax,
        lon_min = args.lonmin,
        lon_max = args.lonmax,
    )
    domain_ds = helpers.extract_patch(domain_position, ds)
    datepart = datetime.strftime(file['Date'], '%Y%m%d_%H_%M')
    filename = f'fnl_{datepart}.nc'

    domain_ds.to_netcdf(
        os.path.join(args.outputdir, filename),
        mode='w', format='NETCDF4')


def main(args=None):
    args = parse_arguments(args)

    assert (args.lon[0] < args.lon[1]) and (args.lat[0] < args.lat[1]), \
            'Invalid latitudes or longitudes.'
    assert os.path.isdir(args.ncep_fnl), \
           f'Invalid input directory: {args.ncep_fnl}'

    files = list_reanalysis_files(args.ncep_fnl)

    # Make output directory.
    os.makedirs(args.outputdir)

    with Pool() as pool:
        tasks = pool.imap_unordered(
            extract_domain,
            (ExtractDomainArgs(
                file=f,
                outputdir=args.outputdir,
                latmin=args.lat[0],
                latmax=args.lat[1],
                lonmin=args.lon[0],
                lonmax=args.lon[1]
            ) for _, f in files.iterrows()))

        # Execute all tasks, and show the progress along the way.
        for _ in tqdm(tasks, total=len(files)):
            pass


if __name__ == '__main__':
    main()
