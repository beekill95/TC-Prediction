"""
Helper for both create_tc_binary_classification* scripts.
"""
from __future__ import annotations


import abc
import cartopy.io.shapereader as shpreader
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
import fiona
import numpy as np
import os
import pandas as pd
import shapely.geometry as sgeom
from shapely.prepared import prep
import time
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


def parse_date_from_nc_filename(filename: str):
    FMT = '%Y%m%d_%H_%M'
    filename, _ = os.path.splitext(os.path.basename(filename))
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, FMT)


def load_best_track(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=(1,))
    # Parse data column.
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')

    # Group by SID, and only retain the first row.
    df = df.groupby('SID', sort=False).first()
    df = df.copy()
    df['SID'] = df.index

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

    ds_lon = ds['lon']
    ds_lat = ds['lat']
    
    lon_cond = (ds_lon >= lon_min) & (ds_lon <= lon_max)
    lat_cond = (ds_lat >= lat_min) & (ds_lat <= lat_max)

    ds = ds.where(lat_cond & lon_cond, drop=True)
    return ds


def is_position_in_dataset(pos: Position, ds: xr.Dataset) -> bool:
    ds_lat = ds['lat'].values
    ds_lon = ds['lon'].values
    lat, lon = pos
    return ((ds_lat.min() < lat < ds_lat.max())
            and (ds_lon.min() < lon < ds_lon.max()))


def is_position_on_ocean(pos: Position) -> bool:
    # TODO: check if this is correct??
    geoms = fiona.open(
                shpreader.natural_earth(
                    resolution='110m', category='physical', name='ocean'))

    ocean_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry'])
                                    for geom in geoms])

    ocean = prep(ocean_geom)

    # Due to Natural Earth's longitude is from -180W to 180E,
    # and our longitude is from 0 to 360, we have to convert
    # to what Natural Earth's longitude.
    lon = pos.lon if pos.lon < 180 else pos.lon - 360

    return ocean.contains(sgeom.Point(lon, pos.lat))


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
        lon = (pos_center.lon + distance * np.cos(rad)) % 360
        center = Position(lat, lon)
        if is_position_in_dataset(center, ds) and is_position_on_ocean(center):
            return center

    raise ValueError('Cannot suggest negative center. Please check your code again!!!')


def neg_output_dir(output_dir: str):
    return os.path.join(output_dir, 'neg')


def pos_output_dir(output_dir: str):
    return os.path.join(output_dir, 'pos')


class PositiveAndNegativePatchesExtractor(abc.ABC):
    max_retries = 3
    seconds_between_retries = 1

    @abc.abstractmethod
    def load_dataset(self, path: str) -> xr.Dataset:
        pass

    def load_dataset_with_retries(self, path: str) -> xr.Dataset:
        i = 0
        while True:
            try:
                ds = self.load_dataset(path)
                return ds
            except Exception as e:
                if i >= self.max_retries:
                    print(f'Give up loading dataset from {path}.')
                    raise e

                time.sleep(self.seconds_between_retries)
                print(f'Retry attempt #{i} - Loading dataset from {path}')
                i += 1


    def __call__(self, args: ExtractPosNegFnArgs) -> None:
        def save_patch(patch: xr.Dataset, center: Position, is_positive: bool):
            fn_parts = [
                # Date.
                datetime.strftime(row['Date'], '%Y%m%d_%H_%M'),
                # Center information.
                f'{center.lat:.1f}_{center.lon:.1f}',
            ]
            if is_positive:
                # Include ID of the storm in the best track,
                # so we can have more information about the storm when needed.
                fn_parts.append(args.row['SID'])

            filename = '_'.join(fn_parts) + '.nc'
            path = os.path.join(
                    args.positive_output_dir
                    if is_positive
                    else args.negative_output_dir, filename)
            patch.to_netcdf(path, mode='w', format='NETCDF4')

        # Unpack arguments
        row = args.row
        domain_size = args.domain_size
        distance = args.distance

        ds = self.load_dataset_with_retries(row['Path'])

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
