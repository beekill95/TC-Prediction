#!/bin/env python3

"""
This script is for detect and track storms candidate from NCEP FNL dataset.
It uses Tempest Extremes' utilities to do so,
specifically, it uses `DetectNodes` to detect storm candidates,
and `StitchNodes` to merge these candidates across different times.
"""

import argparse
from datetime import datetime
import glob
import logging
from multiprocessing import Pool
import os
import subprocess
import tempfile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--indir',
        required=True,
        help='Path to directory contains all .nc files from NCEP FNL dataset.')
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output file')

    return parser.parse_args(args)


def parse_date(nc_file_path: str) -> datetime:
    filename, _ = os.path.splitext(os.path.basename(nc_file_path))
    _, date, hour, minute, *_ = filename.split('_')
    return datetime.strptime(f'{date}_{hour}_{minute}', '%Y%m%d_%H_%M')


def add_time_dimension(nc_file_path: str) -> str:
    """
    Add time dimension to the given nc file
    so that it can be processed properly by TempestExtremes's utilities.
    The time dimension will be parsed from filename.
    It will all so return the path to the new file with time dimension added.
    Noted that the output file will be stored in current directory.
    """
    filename = os.path.basename(nc_file_path)
    logger.info(f'\t ++ Adding time dimension to {filename}')

    # Parse date from file path.
    file_date = parse_date(nc_file_path)
    date_str = file_date.strftime('%Y-%m-%d')
    time_str = file_date.strftime('%H:%M:00')

    add_time_output_file = f'tmp0_{file_date.strftime("%Y%m%d_%H_%M")}.nc'
    set_time_output_file = f'tmp1_{file_date.strftime("%Y%m%d_%H_%M")}.nc'

    # Add time dimension to nc file.
    add_time_cmd = f'module load nco && ncecat -u time {nc_file_path} -O {add_time_output_file}'
    subprocess.run(
        add_time_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        check=True)

    # Set time dimension.
    set_time_cmd = f'module load cdo && cdo -setreftime,{date_str},{time_str} -settaxis,{date_str},{time_str} {add_time_output_file} {set_time_output_file}'
    subprocess.run(
        set_time_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        check=True)

    logger.info(f'\t ++ DONE: Adding time dimension to {filename}')

    return set_time_output_file


def main(args=None):
    arguments = parse_arguments(args)

    # Make sure that we obtain the absolute path so that
    # when we change to temporary directory,
    # we can still work with files users given.
    indir = os.path.abspath(arguments.indir)
    output_path = os.path.abspath(arguments.output)

    # Create temporary directory to store intermediate files.
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f'Changing working directory to {tmpdir} to avoid littering current directory.')

        # Change working directory to `tmpdir` so that
        # Tempest Extremes' utilities' intermediate files can be saved
        # without populating user's working directory.
        os.chdir(tmpdir)

        # Add time dimension to the nc files.
        nc_files = [os.path.join(indir, f) for f in glob.iglob(os.path.join(indir, '*.nc'))]
        nb_files = len(nc_files)
        logger.info(
            f'=== Begin adding time dimension to the given {nb_files} nc files\n'
            '---------------------------------------------------')
        with Pool() as pool:
            nc_files = pool.imap_unordered(add_time_dimension, nc_files)

            # Make sure that the results are ordered.
            nc_files = sorted(nc_files)

        # Save these files to a text file so it can be processed by DetectNodes.
        detect_nodes_inpath = 'DetectNodes_input.txt'
        with open(detect_nodes_inpath, 'w') as detect_nodes_infile:
            lines = [f'{f}\n' for f in nc_files]
            detect_nodes_infile.writelines(lines)

        # Call `DetectNodes` to process these files.
        logger.info(
            '=== Begin detecting low-pressure systems using `DetectNodes`\n'
            '------------------------------------------------------------')
        detect_nodes_out_prefix = 'DetectNodes_output'
        subprocess.run(' '.join([
                'DetectNodes',
                 '--in_data_list', detect_nodes_inpath,
                 '--out', detect_nodes_out_prefix,
                 '--searchbymin pressfc',
                 '--closedcontourcmd "pressfc,100.0,5.5,0"',
                 '--mergedist 6.0',
                 '--regional'
             ]),
            shell=True,
            check=True)

        # Since the above command output in separate text files,
        # we have to put these text files into a list for `StitchNodes`.
        stitch_nodes_inpath = 'StitchNodes_input.txt'
        with open(stitch_nodes_inpath, 'w') as stitch_nodes_infile:
            lines = [f'{detect_nodes_out_prefix}{i:06}.dat\n' for i in range(nb_files)]
            stitch_nodes_infile.writelines(lines)

        # Finally, call `StitchNodes` to process these files.
        # But first, make sure that we can create the output file.
        outputdir = os.path.dirname(output_path)
        os.makedirs(outputdir, exist_ok=True)

        # Now, we can let `StitchNodes` process our files.
        logger.info(
            '\n=== Begin connecting low-pressure systems to form tracks using `StitchNodes`\n'
            '-----------------------------------------------------------------------------')
        subprocess.run(' '.join([
                'StitchNodes',
                '--in_list', stitch_nodes_inpath,
                '--out', output_path,
                '--in_fmt "lon,lat"',
                '--range 8.0',
                '--mintime "54h"',
                '--maxgap "24h"',
                '--threshold "lat,<=,50.0,10;lat,>=,-50.0,10"'
            ]),
            shell=True,
            check=True)



if __name__ == '__main__':
    main()
    logger.info('\n=== DONE ===')
