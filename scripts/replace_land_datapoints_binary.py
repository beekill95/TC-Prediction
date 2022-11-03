#!/bin/env python3
"""
Script for filling land data points with the average of ocean data points.
Only work with binary datasets.
"""
from __future__ import annotations

import argparse
import cartopy.io.shapereader as shpreader
import fiona
from functools import wraps
import glob
from itertools import chain
import os
from multiprocessing import Pool
import numpy as np
import shapely.geometry as sgeom
from shapely.prepared import prep
from tqdm.autonotebook import tqdm
import xarray as xr


def vectorized(otypes):
    def _vectorized_decorator(func):
        vectorized_fn = np.vectorize(func, otypes=otypes)
        return wraps(func)(vectorized_fn)

    return _vectorized_decorator


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('indir', help='Path to input folder.')
    parser.add_argument('outdir', help='Path to output folder.')

    return parser.parse_args(args)


@vectorized([sgeom.Point])
def create_points(lon, lat):
    return sgeom.Point(lon, lat)


@vectorized([bool])
def is_point_on_ocean(ocean, point: sgeom.Point):
    return ocean.contains(point)


def ocean():
    geoms = fiona.open(
        shpreader.natural_earth(
            resolution='110m',
            category='physical',
            name='ocean'))
    ocean_geom = sgeom.MultiPolygon(
        [sgeom.shape(geom['geometry'])
        for geom in geoms])
    ocean = prep(ocean_geom)
    return np.array([ocean])


def generate_ocean_mask(ds: xr.Dataset):
    lat = ds['lat'].values
    lon = ds['lon'].values
    # Convert lon to what to be expected by Natural Earth
    lon = np.where(lon < 180, lon, lon - 360)
    xx, yy = np.meshgrid(lon, lat)
    points = create_points(xx, yy)
    return is_point_on_ocean(ocean(), points)


def replace_land_data_points_with_avg_ocean_value(args: tuple[str, str]):
    def replace(ocean_mask, arr):
        nan_mask = ~np.isfinite(arr)
        land_mask = ~ocean_mask
        masked_array = np.ma.array(arr, mask=nan_mask & land_mask)
        axis = (1, 2) if arr.ndim == 3 else None
        means = masked_array.mean(axis=axis, keepdims=True)
        return np.where(land_mask, means * np.ones_like(arr), arr)

    inpath, outpath = args

    infile = xr.load_dataset(inpath)
    ocean_mask = generate_ocean_mask(infile)

    # TODO:
    variables = {}
    for varname, arr in infile.data_vars.items():
        replaced_arr = replace(ocean_mask, arr.values)
        variables[varname] = arr.copy(data=replaced_arr)

    replaced_ds = infile.copy(data=variables)

    # Save to output path.
    replaced_ds.to_netcdf(outpath, format='NETCDF4')


def main(args=None):
    args = parse_arguments(args)

    # Discover positive and negative files.
    positive_files = glob.iglob(
        os.path.join('pos', '*.nc'), root_dir=args.indir)
    negative_files = glob.iglob(
        os.path.join('neg', '*.nc'), root_dir=args.indir)
    input_files = chain(positive_files, negative_files)

    # Create a list of input and output files.
    input_output_files = list(
        (os.path.join(args.indir, f), os.path.join(args.outdir, f))
        for f in input_files)

    # Create output folders.
    os.makedirs(os.path.join(args.outdir, 'pos'))
    os.makedirs(os.path.join(args.outdir, 'neg'))

    # Processing files.
    with Pool() as pool:
        tasks = pool.imap_unordered(
            replace_land_data_points_with_avg_ocean_value, input_output_files)

        for _ in tqdm(tasks, total=len(input_output_files)):
            pass


if __name__ == '__main__':
    main()
