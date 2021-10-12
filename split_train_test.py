#!/bin/env python3

import argparse
import glob
import os
import pandas as pd
import shutil


def parse_arguments(args=None):
    parser = argparse.ArgumentParser('TC Train Test Split')

    parser.add_argument(
        '--test-from', '-tf',
        dest='test_start',
        action='store',
        required=True,
        help='Test data start from this date. Format: YYYYMMDD.')
    parser.add_argument(
        '--val-from', '-vf',
        dest='val_start',
        action='store',
        help='(Optional) Validation data start from this date. Format: YYYYMMDD.')

    parser.add_argument(
        'indir',
        action='store',
        help='Path to directory contains all observations (.nc files) and groundtruth (tc.csv).')
    parser.add_argument(
        'outdir',
        action='store',
        help='Path to directory contains all outputs, the output directories will be suffixed with _train, _val or _test.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments()

    indir = os.path.join(args.indir, '')
    if not os.path.isdir(indir):
        raise ValueError(f'ERR: Invalid directory {indir}')

    _, indir_name = os.path.split(os.path.dirname(indir))

    # Make sure that validation start date is before than the test data.
    if args.val_start is not None:
        assert args.val_start < args.test_start, f'ERR: Validation date {args.val_start} must be before test date {args.test_start}'

    # List all observation files from the input directory.
    files = glob.glob(os.path.join(args.indir, '*.nc'))
    files.sort()

    tc = pd.read_csv(os.path.join(args.indir, 'tc.csv'),
                     dtype={'Observation': str})

    # Output directories.
    if args.val_start:
        outdirs = [
            ['train', args.val_start],
            ['val', args.test_start],
            ['test', None],
        ]
    else:
        outdirs = [
            ['train', args.test_start],
            ['test', None],
        ]

    # Loop through each output directory and copy files to that.
    ifile = iter(files)
    itc = tc.iterrows()
    for dtype, end_date in outdirs:
        print(f'Splitting data to {dtype} dir')

        # Create output directory
        outdir = os.path.join(args.outdir, f'{indir_name}_{dtype}')
        if os.path.isdir(outdir):
            print(
                f'WARN: {outdir} already exists. Resulting files might not be what you expect!!')
        else:
            os.makedirs(outdir)

        # Filter out files that belongs to this directory.
        file = None
        file = next(ifile) if file is None else file
        while end_date is None or os.path.basename(file).split('_')[1] < end_date:
            # Copy observation to output directory.
            shutil.copy(file, outdir)

            # Advance to the next file.
            try:
                file = next(ifile)
            except StopIteration:
                break

        # Filter out tc belongs to this directory.
        tc_in_type = pd.DataFrame(columns=tc.columns)
        tc_row = None
        _, tc_row = next(itc) if tc_row is None else (None, tc_row)
        while end_date is None or tc_row['Observation'][:8] < end_date:
            tc_in_type = tc_in_type.append(tc_row, ignore_index=True)
            try:
                _, tc_row = next(itc)
            except StopIteration:
                break

        # Then, create tc files.
        tc_in_type.to_csv(os.path.join(outdir, 'tc.csv'))
