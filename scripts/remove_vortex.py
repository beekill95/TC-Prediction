import argparse
from ast import literal_eval
from multiprocessing import Pool
import os
import pandas as pd
from shutil import copyfile
from tc_formation.vortex_removal import vortex_removal as vr
import xarray as xr


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--label', '-l',
            type=str,
            required=True,
            help='Path to .csv label file.',
    )
    parser.add_argument(
            '--output-dir', '-o',
            dest='output_dir',
            type=str,
            required=True,
            help='Path to directory contains output files.'
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
    parser.add_argument(
            '--overwrite',
            type=bool,
            default=False,
            help='Overwrite the contents inside output directory.'
    )

    return parser.parse_args(args)


def remove_vortex_if_necessary(row, output_dir, radius):
    """
    If `row`'s other TC column is not empty,
    this function will apply the vortex removal algorithm and
    write results to `output_dir`.
    Otherwise, this will just copy netcdf file to output directory.

    Returns
    -------
        str
           Path to output file. 
    """
    path = row['Path']
    output_path = os.path.join(output_dir, os.path.basename(path))

    if len(row['Other TC Locations']) == 0:
        copyfile(path, output_path)
    else:
        data = xr.open_dataset(path, engine='netcdf4')
        print(radius)
        data = vr.remove_vortex_ds(data, row['Other TC Locations'], radius)
        data.to_netcdf(output_path)

    return output_path


def process_file(row, output_dir, radius):
    output_path = remove_vortex_if_necessary(row, output_dir, radius)
    row['Path'] = output_path
    return row


if __name__ == '__main__':
    args = parse_arguments()

    # Read .csv label file.
    label = pd.read_csv(args.label)
    label['Other TC Locations'] = label['Other TC Locations'].apply(literal_eval)

    # Make sure that the label file has the required column.
    assert 'Other TC Locations' in label.columns, 'Required label file v4+.'

    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=args.overwrite)

    # Process these files in parallel.
    with Pool(args.processes) as p:
        processed_rows = p.starmap(
                process_file,
                [(row, args.output_dir, args.radius)
                 for _, row in label.iterrows()])

    # Save the output
    df = pd.DataFrame(processed_rows)
    df = df.sort_values('Date')
    df.to_csv(os.path.join(args.output_dir, os.path.basename(args.label)))
