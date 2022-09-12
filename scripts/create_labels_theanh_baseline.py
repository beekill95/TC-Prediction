#!/bin/env python3

"""
This script creates labels that can be used to train deep learning model to
predict whether there will be tropical cyclones or not.
This script only works with The Anh's baseline tracking script outputs:
    * .../Tracking_code/9km/output/baseline/tccount_final_1997.txt
    * .../Tracking_code/9km/output/baseline/tccount_final_1998.txt
    * ...
"""

import argparse
from collections import namedtuple
from datetime import datetime, timedelta
import glob
import logging
import os
import pandas as pd
import re


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nc-dir',
        dest='nc_dir',
        action='store',
        required=True,
        help='Path to directory containing netCDF files.',
    )
    parser.add_argument(
        '--best-track-files',
        dest='best_tracks',
        action='store',
        required=True,
        help='Path to all tc_track files for all years.',
        nargs='+',
    )
    parser.add_argument(
        '--leadtimes',
        action='store',
        nargs='+',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--outdir',
        required=True,
        action='store',
        help='Path to output directory.',
    )

    return parser.parse_args(args)


def convert_to_date(days_since_new_year, year):
    delta = timedelta(days_since_new_year)
    new_year = datetime(year, 1, 1, 0, 0)
    return new_year + delta


def parse_year(file_path):
    filename = os.path.basename(file_path)
    filename, _ = os.path.splitext(filename)
    year_part = filename.split('_')[-1]
    return int(year_part)

def parse_nc_date(nc_file_path):
    filename = os.path.basename(nc_file_path)
    matched = re.match(r'^.*_(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})\.nc$', filename)
    if matched is None:
        raise ValueError(f'Cannot match filename {filename} with predefined pattern.')

    return datetime(
        year=int(matched.group(1)),
        month=int(matched.group(2)),
        day=int(matched.group(3)),
        hour=int(matched.group(4)),
        minute=int(matched.group(5)))


class BestTrack:
    Storm = namedtuple('Storm', ['Id', 'Lat', 'Long', 'Date', 'End'])

    def __init__(self, best_track_file_path):
        self._year = parse_year(best_track_file_path)

        besttrack = pd.read_csv(
            best_track_file_path,
            names=['Year', 'Days', 'StormId', 'Long', 'Lat', 'Pressure (Pa)', 'Wind', 'Unknown'],
            delim_whitespace=True)

        self._storms = self._extract_storms(besttrack)

    @property
    def year(self):
        return self._year

    def has_storm(self, date):
        return date in self._storms

    def get_storm(self, date):
        return self._storms[date]

    def _extract_storms(self, besttrack: pd.DataFrame):
        storms = dict()
        besttrack_gb = besttrack.groupby('StormId')
        for stormid, group in besttrack_gb:
            storm = self._create_storm(stormid, group)
            storms[storm.Date] = storm

        return storms

    def _create_storm(self, stormid, rows):
        first_row = rows.iloc[0]
        occurence_date = convert_to_date(first_row.Days, self.year)
        end_date = convert_to_date(rows.iloc[-1].Days, self.year)
        return self.Storm(
            f'{self.year}_{stormid}',
            first_row.Lat,
            first_row.Long,
            occurence_date,
            end_date,
        )
 

def main():
    DATE_TIME_OUTPUT_FMT = '%Y-%m-%d %H:%M:%S'
    args = parse_arguments()

    # Make output directory if necessary.
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Sort best tracks file.
    best_track_files = iter(sorted(args.best_tracks, key=parse_year))
    best_track = BestTrack(next(best_track_files))

    output = []
    for nc_file in sorted(glob.glob(os.path.join(args.nc_dir, '*.nc'))):
        nc_file = os.path.abspath(nc_file)
        file_date = parse_nc_date(nc_file)

        if file_date.year > best_track.year:
            # Find the suitable best track for the file's year.
            try:
                while best_track.year < file_date.year:
                    best_track = BestTrack(next(best_track_files))
            except StopIteration:
                logging.warning(
                    f'Best track for year {file_date.year} doesn\'t exist.'
                    f' Latest best track\'s year is {best_track.year}.'
                    ' Thus, the remaining observations won\'t be added to the resulting label file.')
                break
        elif file_date.year < best_track.year:
            # This should never happen because both list (nc_file and best_track)
            # are sorted. That means both of them should progress year by year.
            # And file's year should always larger or equal best track's year.
            logging.error('File\'s year should never smaller than best track\' year!')

        if file_date.year == best_track.year:
            for leadtime in args.leadtimes:
                future_date = file_date + timedelta(hours=leadtime)
                tc_will_occur = best_track.has_storm(future_date)
                storm = best_track.get_storm(future_date) if tc_will_occur else None

                # We only record the date only if there will be a TC,
                # or the future date is at 0Z (because the best track only have TC at 0z)
                if tc_will_occur or future_date.hour == 0:
                    output.append({
                        'Date': file_date.strftime(DATE_TIME_OUTPUT_FMT),
                        'TC': tc_will_occur,
                        'TC Id': storm.Id if storm else None,
                        'First Observed': storm.Date if storm else None,
                        'Last Observed': storm.End if storm else None,
                        'Latitude': storm.Lat if storm else 0.,
                        'Longitude': storm.Long if storm else 0.,
                        'First Observed Type': 'TS', # default value.
                        'Will Develop to TC': True, # default value.
                        'Developing Date': storm.Date if storm else None,
                        'Path': nc_file,
                        'Is Other TC Happening': False, # We don't have that info,
                        'Other TC Locations': [],
                    })
        else:
            logging.warning(f'Missing year {file_date.year} in the provided best track.')

    # Save to csv file.
    output = pd.DataFrame(output)
    output_fn = '_'.join(f'{l}h' for l in args.leadtimes)
    output_fn = f'tc_{output_fn}.csv'
    output.to_csv(os.path.join(outdir, output_fn))

    logging.info('\n==== DONE!')


if __name__ == '__main__':
    main()
