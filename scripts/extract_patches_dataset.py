#!/usr/bin/env python3

"""
This script will extract all patches from the full domain.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import glob
from multiprocessing import Pool
import numpy as np
import os
from tqdm.auto import tqdm
import xarray as xr


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'indir',
        help='Path to input directory.')
    parser.add_argument(
        'outdir',
        help='Path to output directory.')
    parser.add_argument(
        '--domain-size',
        dest='domain_size',
        type=int,
        default=30,
        help='Domain size in degrees. Default to 30 degrees.')
    parser.add_argument(
        '--stride',
        type=int,
        default=5,
        help='Stride in degrees. Default to 5 degrees.')
    parser.add_argument(
        '--hours',
        type=int,
        default=[0],
        nargs='+',
        help='Hours to keep. Default to 0h.')
    parser.add_argument(
        '--processes',
        type=int,
        default=8,
        help='Number of parallel processes. Default to 8.')

    return parser.parse_args(args)


ExtractPatchesArgs = namedtuple(
    'ExtractPatchesArgs',
    ['path', 'outdir', 'domain_size', 'stride'])
def extract_patches(args: ExtractPatchesArgs):
    path, domain_size, stride = args.path, args.domain_size, args.stride
    filename, ext = os.path.splitext(os.path.basename(path))

    ds = xr.load_dataset(path, engine='netcdf4')

    lat, lon = ds['lat'].values, ds['lon'].values
    minlat, maxlat = lat.min(), lat.max()
    minlon, maxlon = lon.min(), lon.max()

    for domain_lower_lat in np.arange(minlat, maxlat, stride):
        domain_upper_lat = domain_lower_lat + domain_size

        for domain_lower_lon in np.arange(minlon, maxlon, stride):
            domain_upper_lon = domain_lower_lon + domain_size

            if (domain_upper_lat <= maxlat) and (domain_upper_lon <= maxlon):
                patch = ds.sel(
                    lat=slice(domain_lower_lat, domain_upper_lat),
                    lon=slice(domain_lower_lon, domain_upper_lon))

                # Save patch.
                outfilename = f'{filename}_{domain_lower_lat:.2f}_{domain_lower_lon:.2f}{ext}'
                outpath = os.path.join(args.outdir, outfilename)
                patch.to_netcdf(outpath, format='NETCDF4')


def should_keep_file(path: str, keep_hours: list[int]):
    filename, _ = os.path.splitext(os.path.basename(path))
    hour = int(filename.split('_')[2])
    return hour in keep_hours


def main(args=None):
    args = parse_arguments(args)

    files = glob.glob(os.path.join(args.indir, '*.nc'))
    files = list(filter(lambda f: should_keep_file(f, args.hours), files))
    os.makedirs(args.outdir)

    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            extract_patches,
            (ExtractPatchesArgs(f, args.outdir, args.domain_size, args.stride)
             for f in files))

        for _ in tqdm(tasks, total=len(files)):
            pass


if __name__ == '__main__':
    main()
