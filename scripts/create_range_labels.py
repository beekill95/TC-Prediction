#!/bin/env python3

"""
This script will create a range of label.
This will only work with IBTrACS data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import glob
import os
import pandas as pd
import re


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best-track',
        dest='best_track',
        required=True,
        help='Path to IBTrACS best track data (.csv)')

    parser.add_argument(
        '--reanalysis-data',
        dest='reanalysis_data',
        required=True,
        help='Path to reanalysis data. This will also serve as output directory.')

    parser.add_argument(
        '--time-range',
        dest='time_range',
        type=int,
        default=24,
        help='Time ranges to produce label. Must dividable to 6.')

    parser.add_argument(
        '--exclude-0h',
        dest='exclude_0h',
        action='store_true',
        default=False,
        help='Whether we should exclude 0h in the time range output. Default is False, which means 0h will be included by default.')

    parser.add_argument(
        '--include-no-genesis-class',
        dest='include_no_genesis_class',
        action='store_true',
        default=False,
        help='Whether we should prepend additional class for the case no genesis in the next given hours. Default is False, which means no additional class will be added.')

    return parser.parse_args(args)


def parse_latitudes_longitudes_from_reanalysis_dir(reanalysis_dir):
    config_path = os.path.join(reanalysis_dir, 'conf')

    with open(config_path, 'r') as config_file:
        config = config_file.read()

        lat_matched = re.search(r'LATITUDE=\((\d+) (\d+)\)', config)
        lon_matched = re.search(r'LONGITUDE=\((\d+) (\d+)\)', config)

        assert lat_matched is not None and lon_matched is not None, 'Cannot find Latitude or Longitude in `conf` file.'

        latitude = (float(lat_matched.group(1)), float(lat_matched.group(2)))
        longitude = (float(lon_matched.group(1)), float(lon_matched.group(2)))

        return latitude, longitude


def convert_longitude(longitude):
    """
    This function converts longitude from 0E to -180W to 0E to 360E.
    """
    return longitude if longitude > 0 else 360 + longitude


def load_ibtracs(path: str,
        latitudes: tuple[float, float],
        longitudes: tuple[float, float]) -> pd.DataFrame:
    assert os.path.isfile(path), f'Invalid path to IBTrACS best track: {path}'
    IBTRACS_DATE_FMT = '%Y-%m-%d %H:%M:%S'

    df = pd.read_csv(path, skiprows=(1,))
    df['Date'] = pd.to_datetime(df['ISO_TIME'], format=IBTRACS_DATE_FMT)
    df['Longitude'] = df['LON'].apply(convert_longitude)
    df['Latitude'] = df['LAT']

    lat_mask = (df['Latitude'] >= latitudes[0]) & (df['Latitude'] <= latitudes[1])
    lon_mask = (df['Longitude'] >= longitudes[0]) & (df['Longitude'] <= longitudes[1])

    return df[lat_mask & lon_mask]


def list_reanalysis_files(path: str) -> pd.DataFrame:
    assert os.path.isdir(path), f'Invalid path to reanalysis directory: {path}'
    FILE_DATE_FMT = '%Y%m%d_%H_%M'

    files = glob.iglob(os.path.join(path, '*.nc'))
    df = []
    for file in files:
        abs_path = os.path.abspath(file)
        filename, _ = os.path.splitext(os.path.basename(file))
        datepart = '_'.join(filename.split('_')[1:])
        date = datetime.strptime(datepart, FILE_DATE_FMT)

        df.append(dict(Date=date, Path=abs_path))

    return pd.DataFrame(df, columns=['Date', 'Path']).sort_values(by='Date')


def extract_tc_genesis(ibtracs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    TC Genesis is defined to be first time TC appears in the best track.
    """
    genesis_df = ibtracs.drop_duplicates('SID', keep='first')
    remaining_df = ibtracs.drop(index=genesis_df.index)
    return genesis_df, remaining_df


def create_range_output(
        reanalysis_files: pd.DataFrame,
        ibtracs: pd.DataFrame,
        time_range: int,
        exclude_0h: bool,
        include_no_genesis_class: bool) -> pd.DataFrame:
    assert time_range % 6 == 0, f'Invalid time range, {time_range} must dividable by 6'
    genesis_df, remaining_df = extract_tc_genesis(ibtracs)

    df = []
    for _, reanalysis in reanalysis_files.iterrows():
        genesis_gt = []
        genesis_loc = []
        genesis_sid = []

        reanalysis_date = reanalysis['Date']

        # First, check if we have TC genesis in our time ranges.
        for tidx in range(1 if exclude_0h else 0, time_range // 6 + 1):
            future_date = reanalysis_date + timedelta(hours=tidx * 6)
            matched = genesis_df['Date'] == future_date

            loc = []
            sid = []
            genesis_gt.append(1 if matched.sum() > 0 else 0)
            for _, row in genesis_df[matched].iterrows():
                loc.append((row['Latitude'], row['Longitude']))
                sid.append(row['SID'])

            genesis_loc.append(loc)
            genesis_sid.append(sid)

        # Second, prepend a flag to let the model know if we don't have any TC genesis.
        if include_no_genesis_class:
            genesis_gt = [0 if sum(genesis_gt) > 0 else 1] + genesis_gt

        # Third, check if we have other mature tropical cyclones in current observation.
        matched = remaining_df['Date'] == reanalysis_date
        other_tcs = []
        for _, row in remaining_df[matched].iterrows():
            other_tcs.append((row['SID'], row['Latitude'], row['Longitude']))

        # Finally, store our observations info.
        df.append(dict(
            Date=reanalysis_date,
            Genesis=genesis_gt,
            Genesis_Location=genesis_loc,
            Genesis_SID=genesis_sid,
            Other_TC=other_tcs,
            Path=reanalysis['Path']
        ))

    return pd.DataFrame(df)


def save_groundtruth(gt: pd.DataFrame, *, output_dir: str, time_range: int, exclude_0h: bool, include_no_genesis_class: bool) -> None:
    def create_output_name():
        exclude_0h_info = 'no_0h' if exclude_0h else 'with_0h'
        exclude_genesis_info = 'with_genesis_class' if include_no_genesis_class else 'no_genesis_class'
        time_range_info = f'{time_range}h'

        return f'tc_time_range_{time_range_info}_{exclude_0h_info}_{exclude_genesis_info}.csv'

    output_path = os.path.join(output_dir, create_output_name())
    gt.to_csv(output_path, index=False)


def main(args=None):
    args = parse_arguments(args)

    latitudes, longitudes = parse_latitudes_longitudes_from_reanalysis_dir(args.reanalysis_data)
    ibtracs_df = load_ibtracs(args.best_track, latitudes=latitudes, longitudes=longitudes)
    reanalysis_df = list_reanalysis_files(args.reanalysis_data)
    gt = create_range_output(
            reanalysis_df,
            ibtracs_df,
            time_range=args.time_range,
            exclude_0h=args.exclude_0h,
            include_no_genesis_class=args.include_no_genesis_class)

    save_groundtruth(
        gt,
        output_dir=args.reanalysis_data,
        time_range=args.time_range,
        exclude_0h=args.exclude_0h,
        include_no_genesis_class=args.include_no_genesis_class)


if __name__ == '__main__':
    main()
