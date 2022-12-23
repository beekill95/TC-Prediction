#!/usr/bin/env python3

"""
This script will visualize the prediction made
on each patch and plot the results on a map.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from typing import NamedTuple


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'infile',
        help='Path to .csv file containing prediction for each patch.')
    parser.add_argument(
        'outdir', help='Path to output directory.')
    parser.add_argument(
        '--domain-size',
        dest='domain_size',
        default=30, 
        type=float,
        help='Domain size (in degrees) of each patch. Default to 30 degrees.')
    parser.add_argument(
        '--processes', '-p',
        type=int,
        default=8,
        help='Number of parallel processes to run. Default to 8.')

    return parser.parse_args(args)


VisualizePredictionsArgs = NamedTuple(
    'VisualizePredictionsArgs',
    [('name', str),
     ('patches_df', pd.DataFrame),
     ('domain_size', float),
     ('outdir', str),
     ('threshold', float),
    ])
def visualize_predictions(args: VisualizePredictionsArgs):
    filename, _ = os.path.splitext(args.name)
    outfile = os.path.join(args.outdir, f'{filename}.jpg')

    patches_df = calculate_patch_region(args.patches_df, args.domain_size)

    fig, ax = plt.subplots(figsize=(6, 4))
    bm = draw_map(patches_df, ax)
    xx, yy, votes = convert_prediction_to_votes(patches_df, args.threshold)
    cb = bm.contourf(xx, yy, votes, cmap='Reds')
    fig.colorbar(cb, ax=ax)
    
    fig.tight_layout()
    fig.savefig(outfile, format='jpg')
    plt.close(fig)


def calculate_patch_region(df: pd.DataFrame, domain_size: float):
    df['lat_end'] = df['lat'].apply(lambda l: l + domain_size)
    df['lon_end'] = df['lon'].apply(lambda l: l + domain_size)
    return df


def draw_map(patches_df: pd.DataFrame, ax: plt.Axes):
    minlat, minlon = patches_df['lat'].min(), patches_df['lon'].min(),
    maxlat, maxlon = patches_df['lat_end'].max(), patches_df['lon_end'].max(),

    bm = Basemap(
        llcrnrlat=minlat,
        llcrnrlon=minlon,
        urcrnrlat=maxlat,
        urcrnrlon=maxlon,
        projection='cyl',
        resolution='h',
        ax=ax,
    )

    parallels = np.arange(minlat, maxlat, 10)
    meridians = np.arange(minlon, maxlon, 20)
    bm.drawparallels(
        parallels,
        labels=[1, 0, 0, 0],
        color="grey")
    bm.drawmeridians(
        meridians,
        labels=[0, 0, 0, 1],
        color="grey")

    bm.drawcoastlines()
    bm.drawstates()
    bm.drawcountries()
    bm.drawlsmask(land_color='Linen', ocean_color='#CCFFFF')
    bm.drawcounties()

    return bm


def convert_prediction_to_votes(patches_df: pd.DataFrame, threshold: float):
    minlat, minlon = patches_df['lat'].min(), patches_df['lon'].min(),
    maxlat, maxlon = patches_df['lat_end'].max(), patches_df['lon_end'].max(),

    lat = np.linspace(minlat, maxlat, 1000)
    lon = np.linspace(minlon, maxlon, 1000)

    xx, yy = np.meshgrid(lon, lat)
    votes = np.zeros_like(xx)

    for _, r in patches_df.iterrows():
        xx_mask = (xx >= r['lon']) & (xx <= r['lon_end'])
        yy_mask = (yy >= r['lat']) & (yy <= r['lat_end'])
        votes[xx_mask & yy_mask] += (1 if r['pred'] >= threshold else 0)

    return xx, yy, votes


def main(args=None):
    args = parse_arguments(args)

    # Check arguments.
    infile, outdir = args.infile, args.outdir
    assert os.path.isfile(infile), f'Invalid file: {infile}'
    assert not os.path.isdir(outdir), f'Output directory "{outdir}" exists!'

    # Load .csv file.
    pred_df = pd.read_csv(infile)
    pred_df = pred_df.groupby('path')

    # Create output directory.
    os.makedirs(outdir)

    # Visualize predictions in parallel.
    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            visualize_predictions,
            (VisualizePredictionsArgs(
                name=str(name),
                patches_df=df,
                domain_size=args.domain_size,
                outdir=outdir,
                threshold=0.5)
             for name, df in pred_df))

        for _ in tqdm(tasks, total=len(pred_df), desc='Visualizing'):
            pass


if __name__ == '__main__':
    main()
