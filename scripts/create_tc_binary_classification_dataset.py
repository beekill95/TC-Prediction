#!/bin/env python3

"""
This script will create binary classification dataset for tropical cyclogenesis.
"""

from __future__ import annotations


import argparse
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
import glob
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import xarray as xr


Position = namedtuple('Center', ['lat', 'lon'])
PatchPosition = namedtuple(
    'PatchPosition', ['lat_min', 'lat_max', 'lon_min', 'lon_max'])


@dataclass
class ExtractPosNegFnArgs():
    row: pd.Series
    domain_size: float
    distance: float
    output_dir: str

    @property
    def negative_output_dir(self):
        return neg_output_dir(self.output_dir)

    @property
    def positive_output_dir(self):
        return pos_output_dir(self.output_dir)


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
        '--distance',
        default=30,
        type=float,
        help='Distance (in degrees) of the negative domain\'s center to positive domain\'s center. Default is 30deg.')
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output directory.')

    return parser.parse_args(args)


def parse_date_from_nc_filename(filename: str):
    FMT = '%Y%m%d_%H_%M'
    filename, _ = os.path.splitext(os.path.basename(filename))
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, FMT)


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '*.nc'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths
    })


def load_best_track(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=(1,))
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')
    return df


def suggest_patch_position(center: Position, ds: xr.Dataset, domain_size: float) -> PatchPosition:
    """
    This will suggest suitable patch position to be extracted.
    First, it will calculate the normal position (which is the 'center' is exactly at the center),
    and then check if the position is in the domain.
    If it's not, then it will readjust the latitude and then the longitude.
    """
    def adjust(pmin, pmax, domain_min, domain_max):
        if pmin < domain_min:
            diff = domain_min - pmin
            return pmin + diff, pmax + diff
        elif pmax > domain_max:
            diff = pmax - domain_max
            return pmin - diff, pmax - diff

        return pmin, pmax

    half_size = domain_size / 2

    # Latitude.
    lat_min = center.lat - half_size
    lat_max = center.lat + half_size
    lat_min, lat_max = adjust(
        lat_min, lat_max, ds['lat'].min().values, ds['lat'].max().values)

    # Longitude.
    lon_min = center.lon - half_size
    lon_max = center.lon + half_size
    lon_min, lon_max = adjust(
        lon_min, lon_max, ds['lon'].min().values, ds['lon'].max().values)

    return PatchPosition(lat_min, lat_max, lon_min, lon_max)


def extract_patch(pos: PatchPosition, ds: xr.Dataset) -> xr.Dataset:
    lat_min, lat_max, lon_min, lon_max = pos

    # ds = ds.where(
    #     (lat_min <= ds['lat']) & (ds['lat'] <= lat_max)
    #           & (lon_min <= ds['lon']) & (ds['lon'] <= lon_max), drop=True)
    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return ds


def is_position_in_dataset(pos: Position, ds: xr.Dataset) -> bool:
    lat, lon = pos
    return ((ds['lat'].min() < lat < ds['lat'].max())
            and (ds['lon'].min() < lon < ds['lon'].max()))


def suggest_negative_patch_center(pos_center: Position, distance: float, ds: xr.Dataset) -> Position:
    """
    Suggest suitable negative patch's center that is |distance| away from the positive center.
    It will search in 8 directions in counter-clockwise:
        East -> NE -> North -> NW -> West -> SW -> South -> SE
    It will stop searching when the following condition is met:
        The center is in the given domain.
    """
    directions = range(0, 360, 45)
    for angle in directions:
        rad = angle * np.pi / 180
        lat = pos_center.lat + distance * np.sin(rad)
        lon = pos_center.lon + distance * np.cos(rad)
        center = Position(lat, lon)
        if is_position_in_dataset(center, ds):
            return center

    raise ValueError('Cannot suggest negative center. Please check your code again!!!')


def neg_output_dir(output_dir: str):
    return os.path.join(output_dir, 'neg')


def pos_output_dir(output_dir: str):
    return os.path.join(output_dir, 'pos')


def extract_positive_and_negative_patches(args: ExtractPosNegFnArgs) -> None:
    def save_patch(patch: xr.Dataset, center: Position, is_positive: bool):
        date_part = datetime.strftime(row['Date'], '%Y%m%d_%H_%M')
        center_part = f'{center.lat:.1f}_{center.lon:.1f}'
        filename = f'{date_part}_{center_part}.nc'
        path = os.path.join(
                args.positive_output_dir
                if is_positive
                else args.negative_output_dir, filename)
        patch.to_netcdf(path, mode='w', format='NETCDF4')


    # Unpack arguments
    row = args.row
    domain_size = args.domain_size
    distance = args.distance

    ds = xr.load_dataset(row['Path'], engine='netcdf4')

    # Obtain TC location.
    lat = row['LAT']
    lon = row['LON']
    lon = lon if lon > 0 else 360 + lon
    pos_center = Position(lat=lat, lon=lon)

    # Make sure that the TC location is in our dataset's domain.
    if not is_position_in_dataset(pos_center, ds):
        return

    # Extract positive patch.
    pos_patch_pos = suggest_patch_position(pos_center, ds, domain_size)
    pos_patch = extract_patch(pos_patch_pos, ds)

    # Extract suitable negative patch.
    neg_center = suggest_negative_patch_center(pos_center, distance, ds)
    neg_patch_pos = suggest_patch_position(neg_center, ds, domain_size)
    neg_patch = extract_patch(neg_patch_pos, ds)

    # Save both patches.
    save_patch(pos_patch, pos_center, True)
    save_patch(neg_patch, neg_center, False)

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
            extract_positive_and_negative_patches,
            (ExtractPosNegFnArgs(row, args.domain_size, args.distance, args.output)
             for _, row in best_track.iterrows()))

        # Loop through tasks so they get executed.
        # Also, show the progress along the way.
        for _ in tqdm(tasks, total=len(best_track)):
            pass

if __name__ == '__main__':
    main()
