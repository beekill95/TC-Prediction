#!/usr/bin/env python3

"""
This script will remove the developed storms from the domain (NCEP/FNL).
The developed storms are storms that are not TC genesis
as indicated by the first entry in the ibtracs best track.
"""

import argparse
from collections import namedtuple
from datetime import datetime
import glob
from multiprocessing import Pool
import os
import pandas as pd
from shutil import copyfile
import tc_formation.vortex_removal.vortex_removal as vr
from tqdm import tqdm
import xarray as xr


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'indir',
        help='Path to the directory containing NCEP/FNL .nc files.'
    )
    parser.add_argument(
        'outdir',
        help='Path to directory containing output files.'
    )
    parser.add_argument(
        '--ibtracs',
        type=str,
        required=True,
        help='Path to ibtracs best track file.',
    )
    parser.add_argument(
        '--radius', '-r',
        type=float,
        default=10.0,
        help='Radius of tropical cyclones region to apply removal algorithm.',
    )
    parser.add_argument(
        '--processes', '-p',
        type=int,
        default=4,
        help='Number of parallel processes to use.',
    )

    return parser.parse_args(args)


DevelopedStormsRemovalArgs = namedtuple(
    'DevelopedStormsRemovalArgs',
    ['filepath', 'developed_storms_locations', 'storm_radius', 'outdir'])
def remove_developed_storms_if_necessary(args: DevelopedStormsRemovalArgs):
    path = args.filepath
    output_path = os.path.join(args.outdir, os.path.basename(path))

    storms_locations = args.developed_storms_locations
    if len(storms_locations) == 0:
        # If there is not developed storms in the domain,
        # then just copy the file to the output folder.
        copyfile(path, output_path)
    else:
        data = xr.open_dataset(path, engine='netcdf4')
        data = vr.remove_vortex_ds(data, storms_locations, args.storm_radius)
        data.to_netcdf(output_path)

    return output_path


def find_developed_storms(files_df: pd.DataFrame, developed_storms_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, file_row in files_df.iterrows():
        storms_in_the_day = developed_storms_df[developed_storms_df['Date'] == file_row['Date']]
        developed_storms_locations = [
            (row['Lat'], row['Lon']) for _, row in storms_in_the_day.iterrows()
        ]
        results.append({
            'Path': file_row['Path'],
            'Storms Locations': developed_storms_locations,
        })

    return pd.DataFrame(results)


def list_reanalysis_files(path: str) -> pd.DataFrame:
    def parse_date_from_nc_filename(filename: str):
        FMT = '%Y%m%d_%H_%M'
        filename, _ = os.path.splitext(os.path.basename(filename))
        datepart = '_'.join(filename.split('_')[1:])
        return datetime.strptime(datepart, FMT)

    files = glob.iglob(os.path.join(path, '*.nc'))
    files = ((parse_date_from_nc_filename(f), f) for f in files)
    dates, filepaths = zip(*files)
    return pd.DataFrame({
        'Date': dates,
        'Path': filepaths,
    })


def load_developed_storms_from_ibtracs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=(1,), na_filter=False)
    # Parse data column.
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')

    # We only care about some columns.
    df = df[['SID', 'Date', 'LAT', 'LON']]
    df = df.rename(columns=dict(LAT='Lat', LON='Lon'))

    # Convert LON.
    df['Lon'] = df['Lon'].apply(lambda l: l if l > 0 else 360 + l)

    # Group by SID, and only retain the first row.
    genesis_df = df.groupby('SID', sort=False).first()

    # Filter out all the genesis events.
    df_index = pd.MultiIndex.from_arrays([df['SID'], df['Date']])
    genesis_index = pd.MultiIndex.from_arrays([genesis_df.index, genesis_df['Date']])
    genesis_mask = df_index.isin(genesis_index)
    return df[~genesis_mask]


def filter_storms_not_in_domain(files_df: pd.DataFrame, developed_storms_df: pd.DataFrame) -> pd.DataFrame:
    ds = xr.load_dataset(files_df['Path'].iloc[0], engine='netcdf4')
    latmin, latmax = tuple(fn(ds['lat'].values) for fn in [min, max])
    lonmin, lonmax = tuple(fn(ds['lon'].values) for fn in [min, max])
    
    lat_mask = (latmin <= developed_storms_df['Lat']) & (developed_storms_df['Lat'] <= latmax)
    lon_mask = (lonmin <= developed_storms_df['Lon']) & (developed_storms_df['Lon'] <= lonmax)
    return developed_storms_df[lat_mask & lon_mask]


def main(args=None):
    args = parse_arguments(args)

    files_df = list_reanalysis_files(args.indir)
    developed_storms_df = load_developed_storms_from_ibtracs(args.ibtracs)
    developed_storms_df = filter_storms_not_in_domain(files_df, developed_storms_df)
    files_with_developed_storms_df = find_developed_storms(files_df, developed_storms_df)

    # Create output directory.
    os.makedirs(args.outdir)

    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            remove_developed_storms_if_necessary,
            [DevelopedStormsRemovalArgs(row['Path'], row['Storms Locations'], args.radius, args.outdir)
             for _, row in files_with_developed_storms_df.iterrows()])

        for _ in tqdm(tasks, total=len(files_with_developed_storms_df), desc='Removing Vortex'):
            pass


if __name__ == '__main__':
    main()
