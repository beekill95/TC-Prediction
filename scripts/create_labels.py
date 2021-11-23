#!/bin/env python3

import argparse
from datetime import datetime, timedelta
import glob
import os
import pandas as pd
import re
from typing import Tuple, List


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best-track',
        dest='best_track',
        action='store',
        required=True,
        help='''
        Path to directory contains best track data from JTWC.
        The directory should contain the best track for all storms:
        best_track_folder/
        ├─ bwp012008.dat
        ├─ bwp022008.dat
        ├─ ...
        ├─ bwp012009.dat
        ├─ ...

        Or ibtracs .csv file.
        ''')

    parser.add_argument(
        '--best-track-from',
        dest='best_track_from',
        action='store',
        choices=['jtwc', 'ibtracs'],
        required=True,
        help='Where the best track data is from.')

    parser.add_argument(
        '--observations-dir',
        dest='observations_dir',
        action='store',
        required=True,
        help='''
            Path to directory contains all observation files .nc and .conf file.
            This is also the directory that the will contain the output csv file.
            The output file will be `tc_{lead_time}h.csv`.
            ''')

    parser.add_argument(
        '--leadtime',
        nargs='+',
        dest='leadtime',
        action='store',
        type=int,
        default=[12],
        help='The lead time to generate the data. Default is 12h.')

    parser.add_argument(
        '--basins',
        nargs='+',
        action='store',
        default=['WP', 'EP'],
        help='The basins that we should generate labels.')

    parser.add_argument(
        '--nature',
        nargs='+',
        action='store',
        default=['DS', 'TC'],
        help='Only include best tracks from these storms type.')

    return parser.parse_args(args)


def parse_lat_long_from_config(config_path):
    with open(config_path, 'r') as config_file:
        config = config_file.read()

        lat_matched = re.search(r'LATITUDE=\((\d+) (\d+)\)', config)
        lon_matched = re.search(r'LONGITUDE=\((\d+) (\d+)\)', config)

        assert lat_matched is not None and lon_matched is not None, 'Cannot find Latitude or Longitude in `conf` file.'

        latitude = (int(lat_matched.group(1)), int(lat_matched.group(2)))
        longitude = (int(lon_matched.group(1)), int(lon_matched.group(2)))

        return latitude, longitude

def parse_date_from_observation_filename(filename: str):
    filename, _ = os.path.splitext(os.path.basename(filename))
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, '%Y%m%d_%H_%M')


def get_date_range_of_observations(observations_dir, best_track_year_start, best_track_year_end):
    observations = glob.glob(os.path.join(observations_dir, '*.nc'))
    observations = sorted(observations)
    observations = map(parse_date_from_observation_filename, observations)
    observations = filter(lambda date: best_track_year_start <=
                          date <= best_track_year_end, observations)
    observations = list(observations)
    return observations[0], observations[-1]


def get_best_track_year_range_jtwc(best_track_folder, basins: List[str]):
    best_tracks = sum(
            (glob.glob(os.path.join(best_track_folder, f'b{basin.lower()}*'))
                for basin in basins),
            [])
    best_tracks = map(os.path.basename, best_tracks)
    years = sorted(list(set(''.join(str(fn)[5:9]) for fn in best_tracks)))
    return datetime(int(years[0]), 1, 1), datetime(int(years[-1]), 12, 31)

def get_best_track_year_range_ibtracs(best_track_path: str, basins: List[str]):
    best_track = pd.read_csv(best_track_path)
    best_track = best_track[best_track['BASIN'].isin([b.upper() for b in basins])]
    dates = pd.to_datetime(best_track['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')
    return dates.iloc[0], dates.iloc[-1]

def load_best_track(best_track_path) -> pd.DataFrame:
    """
    Load best track .dat file as pandas Dataframe.
    Documentation of all the columns' meaning is on: https://www.metoc.navy.mil/jtwc/jtwc.html?western-pacific
    """
    def convert_latitude(latitude: str):
        value = float(''.join(list(latitude)[:-1]))
        return (value / 10) * (1 if latitude.endswith('N') else -1)

    def convert_longitude(longitude: str):
        value = float(''.join(list(longitude)[:-1]))
        return value / 10 if longitude.endswith('E') else 360 - value / 10

    columns = [
        "BASIN" , "CY" , "YYYYMMDDHH" , "TECHNUM" , "TECH" , "TAU" , "LatN/S" , "LonE/W" , "VMAX" , "MSLP" ,
        "TY" , "RAD" , "WINDCODE" , "RAD1" , "RAD2" , "RAD3" , "RAD4" , "RADP" , "RRP" , "MRD" , "GUSTS" , "EYE" ,
        "SUBREGION" , "MAXSEAS" , "INITIALS" , "DIR" , "SPEED" , "STORMNAME" , "DEPTH" , "SEAS" ,
        "SEASCODE" , "SEAS1" , "SEAS2" , "SEAS3" , "SEAS4"
    ]
        
    df = pd.read_csv(best_track_path, names=columns, delimiter=',', index_col=False, skipinitialspace=True)
    df['LatN/S'] = df['LatN/S'].apply(convert_latitude)
    df['LonE/W'] = df['LonE/W'].apply(convert_longitude)

    return df

def filter_tc_in_domain(best_track: pd.DataFrame, latitude: Tuple[int, int], longitude: Tuple[int, int]):
    in_latitude = (best_track['Latitude'] >= latitude[0]) & (best_track['Latitude'] <= latitude[1])
    in_longitude = (best_track['Longitude'] >= longitude[0]) & (best_track['Longitude'] <= longitude[1])
    return best_track[in_latitude & in_longitude]


# TODO: right now, I will just use the first row of the best track,
# and I will consider it as tropical cyclones,
# even though it's just a tropical disturbances, and might not evolve into tropical cyclones later.
def extract_tropical_cyclones_from_jtwc_best_track(best_track_folder: str, latitude: Tuple[int, int], longitude: Tuple[int, int], basins: List[str]):
    def extract_tc_information(tc_id: str, best_track: pd.DataFrame):
        def parse_date(date):
            return datetime.strptime(str(date), '%Y%m%d%H')

        first_row = best_track.iloc[0]
        last_row = best_track.iloc[-1]

        # The time that this disturbance is a tropical cyclone.
        tc_period = best_track[best_track['TY'].isin(['TC', 'TS'])]

        return {
            'Id': tc_id,
            'Latitude': first_row['LatN/S'],
            'Longitude': first_row['LonE/W'],
            'First Observed': parse_date(first_row['YYYYMMDDHH']),
            'Last Observed': parse_date(last_row['YYYYMMDDHH']),
            'First Observed Type': first_row['TY'],
            'Developing to TC': len(tc_period) > 0,
            'Developing Date': parse_date(tc_period.iloc[0]['YYYYMMDDHH']) if len(tc_period) > 0 else None
        }

def extract_tropical_cyclones_from_ibtracs_best_track(
        best_track_path: str,
        latitude: Tuple[int, int],
        longitude: Tuple[int, int],
        basins: List[str]):
    def convert_latitude(latitude):
        # FIXME:
        # Right now, just leave it as is.
        # As we are currently dealing with Pacific Ocean only.
        return latitude

    def convert_longitude(longitude):
        # East will be positive from 0 to 180E
        # West will be negative from 0 to -180W
        # All these will be converted to 0 to 360,
        # with 0 - 180 belongs to East, and 180 to 360 belongs to West
        return longitude if longitude > 0 else 360 + longitude

    def extract_tc_information(df):
        tc_period = df[df['NATURE'].isin(['TC', 'TS'])]

        first_row = df.iloc[0]
        last_row = df.iloc[-1]
    

        return {
            'Id': first_row['SID'],
            'Latitude': convert_latitude(first_row['LAT']),
            'Longitude': convert_longitude(first_row['LON']),
            'First Observed': first_row['ISO_TIME'],
            'Last Observed': last_row['ISO_TIME'],
            'First Observed Type': first_row['NATURE'],
            'Developing to TC': len(tc_period) > 0,
            'Developing Date': tc_period['ISO_TIME'].iloc[0] if len(tc_period) > 0 else None
        }

    tc_df = pd.read_csv(best_track_path)

    # We only need tropical cyclones within these basins.
    tc_df = tc_df[tc_df['BASIN'].isin([b.upper() for b in basins])]
    tc_df['ISO_TIME'] = pd.to_datetime(tc_df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')
    
    # Then, we will group by SID so we can process each storm one by one.
    tc = []
    for _, group in tc_df.groupby('SID'):
        tc.append(extract_tc_information(group))

    return pd.DataFrame(tc)


def create_labels(
        observations_dir: str,
        tc: pd.DataFrame,
        observation_ranges: Tuple[datetime, datetime],
        leadtimes: List[int]):
    labels = []

    for observation_filename in glob.iglob(os.path.join(observations_dir, '*.nc')):
        observation_date = parse_date_from_observation_filename(observation_filename)
        if not (observation_ranges[0] <= observation_date <= observation_ranges[1]):
            continue

        has_tropical_cnt = 0

        for leadtime in leadtimes:
            next_leadtime = observation_date + timedelta(hours=leadtime)
            has_tropical = tc['First Observed'] == next_leadtime
            has_tropical_cnt += len(tc[has_tropical])

            for _, tc_row in tc[has_tropical].iterrows():
                observation_label = {
                    'Date': observation_date,
                    'TC': True,
                    'TC Id': tc_row['Id'],
                    'First Observed': tc_row['First Observed'],
                    'Last Observed': tc_row['Last Observed'],
                    'Latitude': tc_row['Latitude'],
                    'Longitude': tc_row['Longitude'],
                    'First Observed Type': tc_row['First Observed Type'],
                    'Will Develop to TC': tc_row['Developing to TC'],
                    'Developing Date': tc_row['Developing Date'],
                    'Path': observation_filename,
                }
                labels.append(observation_label)

        if has_tropical_cnt == 0:
            is_tc_occuring = (tc['First Observed'] <= observation_date) & (tc['Last Observed'] >= observation_date)
            if len(tc[is_tc_occuring]) == 0:
                labels.append({
                    'Date': observation_date,
                    'TC': False,
                    'Path': observation_filename,
                })

    labels = pd.DataFrame(
            labels,
            columns=[
                'Date', 'TC', 'TC Id',
                'First Observed', 'Last Observed',
                'Latitude', 'Longitude',
                'First Observed Type', 'Will Develop to TC', 'Developing Date',
                'Path'])
    labels.sort_values(by='Date', inplace=True, ignore_index=True)
    return labels


if __name__ == '__main__':
    args = parse_arguments()

    latitude, longitude = parse_lat_long_from_config(
        os.path.join(args.observations_dir, 'conf'))
    print(
        f'Will create labels with storms in latitude {latitude} and longitude {longitude}.')

    if args.best_track_from == 'jtwc':
        best_track_year_range = get_best_track_year_range_jtwc(args.best_track, args.basins)
    else:
        best_track_year_range = get_best_track_year_range_ibtracs(args.best_track, args.basins)

    observation_start_date, observation_end_date = get_date_range_of_observations(
        args.observations_dir,
        *best_track_year_range)
    print(f'Only create labels for observations from {observation_start_date} to {observation_end_date}.')

    # Extract tropical cyclones
    if args.best_track_from == 'jtwc':
        tc_df = extract_tropical_cyclones_from_jtwc_best_track(
            args.best_track, latitude, longitude, args.basins)
    else:
        tc_df = extract_tropical_cyclones_from_ibtracs_best_track(
            args.best_track, latitude, longitude, args.basins)

    # After that, filter out all tropical cyclones that are not in our domain of interest.
    tc_df = filter_tc_in_domain(tc_df, latitude, longitude)

    # Then, base on lead time to create labels for each observation date.
    tc_df.sort_values(by='First Observed', inplace=True, ignore_index=True)

    labels = create_labels(
        observations_dir=args.observations_dir,
        tc=tc_df,
        observation_ranges=(observation_start_date, observation_end_date),
        leadtimes=args.leadtime,
    )
    leadtime_str = '_'.join(f'{l}h' for l in args.leadtime)
    labels.to_csv(
        os.path.join(args.observations_dir, f'tc_{args.best_track_from}_{leadtime_str}.csv'),
        index=False
    )
