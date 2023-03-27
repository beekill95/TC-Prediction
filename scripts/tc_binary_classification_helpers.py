"""
Helper for both create_tc_binary_classification* scripts.
"""
from __future__ import annotations


import abc
import cartopy.io.shapereader as shpreader
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
import fiona
import glob
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
    distances: list[float]
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


def load_best_track(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, skiprows=(1,), na_filter=False)
    # Parse data column.
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')

    # We only care about some columns.
    df = df[['SID', 'Date', 'LAT', 'LON', 'BASIN']]

    # Convert LON.
    df['LON'] = df['LON'].apply(lambda l: l if l > 0 else 360 + l)

    # Group by SID, and only retain the first row.
    genesis_df = df.groupby('SID', sort=False).first()
    genesis_df = genesis_df.copy()
    genesis_df['SID'] = genesis_df.index

    return genesis_df, df


def load_best_track_files_theanh(files_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def convert_to_date(days_since_new_year, year):
        # Base on what I found,
        # 121 corresponds to May 1st
        # 153 corresponds to Jun 2nd
        # So, in order to get that, we have to minus 1 from the days_since_new_year.
        delta = timedelta(days_since_new_year - 1)
        new_year = datetime(year, 1, 1, 0, 0)
        return new_year + delta

    def parse_year_from_dir(file_path):
        parent_dir = os.path.dirname(file_path).split(os.path.sep)[-1]
        year_part = parent_dir.split('_')[-1]
        return int(year_part)

    def parse_year_from_file(file_path):
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        year_part = name.split('_')[-1]
        return int(year_part)

    files = glob.iglob(files_pattern)

    storms = []
    for file in files:
        try:
            year = parse_year_from_dir(file)
        except ValueError:
            year = parse_year_from_file(file)

        storms_in_year = pd.read_csv(
            file,
            names=['Days', 'StormId', 'LON', 'LAT'],
            delim_whitespace=True,
            usecols=list(range(4)),
        )
        storms_in_year['SID'] = storms_in_year['StormId'].apply(
            lambda id: f'{year}-{id}')

        # Convert 'Days' in year to date.
        storms_in_year['Date'] = storms_in_year['Days'].apply(
            lambda days: convert_to_date(days, year))

        storms.append(
            storms_in_year[['SID', 'Date', 'LAT', 'LON']])

    storms_df = pd.concat(storms).sort_values('Date')

    genesis_df = storms_df.groupby('SID').first().copy()
    genesis_df['SID'] = genesis_df.index
    return genesis_df, storms_df


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


def suggest_negative_patch_center(pos_center: Position, distances: list[float], ds: xr.Dataset) -> Position:
    """
    Suggest suitable negative patch's center that is |distance| away from the positive center.
    It will search in 8 directions in counter-clockwise:
        East -> NE -> North -> NW -> West -> SW -> South -> SE
    It will stop searching when the following condition is met:
        The center is in the given domain.
    """
    directions = range(0, 360, 45)
    distances = [*distances]
    distances.sort(reverse=True)

    for distance in distances:
        for angle in directions:
            rad = angle * np.pi / 180
            lat = pos_center.lat + distance * np.sin(rad)
            lon = (pos_center.lon + distance * np.cos(rad)) % 360
            center = Position(lat, lon)
            if is_position_in_dataset(center, ds) and is_position_on_ocean(center):
                # return center
                yield center

        # DEBUG
        # print(f'lat {ds["lat"].values.min()} - {ds["lat"].values.max()}')
        # print(f'lon {ds["lon"].values.min()} - {ds["lon"].values.max()}')
        # print(f'{pos_center=}')
        # print(f'{center=}, {is_position_in_dataset(center, ds)=}, {is_position_on_ocean(center)=}')

    raise ValueError('Cannot suggest negative center. Please check your code again!!!')


def does_patch_contain_TC(date: datetime, patch: PatchPosition, best_track: pd.DataFrame) -> bool:
    lat_min = patch.lat_min
    lat_max = patch.lat_max
    lon_min = patch.lon_min
    lon_max = patch.lon_max
    
    best_track = best_track[best_track['Date'] == date]
    tc_lon = best_track['LON']
    tc_lat = best_track['LAT']

    tc_in_domain = best_track[
        (tc_lon >= lon_min)
        & (tc_lon <= lon_max)
        & (tc_lat >= lat_min)
        & (tc_lat <= lat_max)]

    return len(tc_in_domain) > 0


def neg_output_dir(output_dir: str):
    return os.path.join(output_dir, 'neg')


def pos_output_dir(output_dir: str):
    return os.path.join(output_dir, 'pos')


class PositiveAndNegativePatchesExtractor(abc.ABC):
    max_retries = 3
    seconds_between_retries = 1

    def __init__(self,
            detailed_best_track: pd.DataFrame,
            raise_cannot_find_negative_patch: bool = True) -> None:
        self.raise_cannot_find_negative_patch = raise_cannot_find_negative_patch
        self.detailed_best_track = detailed_best_track

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
            # Make sure that the patch is in correct order.
            patch = patch.reindex(
                lat=sorted(patch['lat']),
                lon=sorted(patch['lon']),
                lev=sorted(patch['lev'], reverse=True))

            fn_parts = [
                # Date.
                datetime.strftime(row['OriginalDate'], '%Y%m%d_%H_%M'),
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
        distances = args.distances

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
        save_patch(pos_patch, pos_center, True)

        # Extract suitable negative patch.
        try:
            for neg_center in suggest_negative_patch_center(pos_center, distances, ds):
                neg_patch_pos = suggest_patch_position(neg_center, ds, domain_size)
                if not does_patch_contain_TC(row['OriginalDate'], neg_patch_pos, self.detailed_best_track):
                    neg_patch = extract_patch(neg_patch_pos, ds)
                    save_patch(neg_patch, neg_center, False)
                    break

        except ValueError as e:
            if self.raise_cannot_find_negative_patch:
                raise e
            else:
                print(f'Ignore generating negative patch for file {row["Path"]}.')


def list_reanalysis_files(path: str) -> pd.DataFrame:
    files = glob.iglob(os.path.join(path, '*.nc'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths,
    })


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
def extract_all_variables(ds: xr.Dataset, order: list[str]):
    values = []

    for varname in order:
        var = ds[varname].values
        if var.ndim == 2:
            var = var[None, ...]

        values.append(var)

    values = np.concatenate(values, axis=0)
    values = np.moveaxis(values, 0, 2)
    return values


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
