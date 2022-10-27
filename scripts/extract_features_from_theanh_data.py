#!/bin/env python3

"""
This script will extract environmental variables from The Anh's data,
which is WRF model's output. The list of variables to be extracted are:
    * Absolute vorticity
    * Relative humidity
    * Temperature
    * Geopotential height
    * Vertical Velocity
    * U-wind and V-wind
    * Cape
    * Surface Temperature
    * Surface Pressure

In addition, to be compatible with the original data (NCEP FNL reanalsysi),
the script will also extract the domain from 5 degree North to 45 degree North,
and 100 degree West to 260 degree East,
and vertical pressure levels are 19 mandatory levels
(1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200).
"""
from __future__ import annotations


import argparse
from collections import OrderedDict, namedtuple
import logging
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.typing as npt
from multiprocessing import Pool
import os
from tqdm import tqdm
from typing import Callable
import wrf
import xarray as xr

PRESSURE_LEVELS = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
]
TIME_FORMAT = '%Y%m%d_%H_%M'

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'outputdir',
        action='store',
        help='Path to the output director.',
    )
    parser.add_argument(
        'inputfiles',
        nargs='+',
        action='store',
        help='Path to the input file.',
    )
    parser.add_argument(
        '--prefix', '-p',
        action='store',
        required=True,
        help='Prefix to be prepended to the output filename.'
    )

    return parser.parse_args(args)


def calculate_vorticity(ds: Dataset):
    south_north_size = ds.dimensions['south_north'].size
    west_east_size = ds.dimensions['west_east'].size

    # Fill in missing variables so we can calculate vorticity.
    mapfac_u = ds.createVariable('MAPFAC_U', 'f4', ('Time', 'south_north', 'west_east_stag'))
    mapfac_u[:, :, :] = np.ones(
        (1, south_north_size, west_east_size + 1), dtype=np.float32)

    mapfac_v = ds.createVariable('MAPFAC_V', 'f4', ('Time', 'south_north_stag', 'west_east'))
    mapfac_v[:, :, :] = np.ones(
        (1, south_north_size + 1, west_east_size), dtype=np.float32)

    mapfac_m = ds.createVariable('MAPFAC_M', 'f4', ('Time', 'south_north', 'west_east'))
    mapfac_m[:, :, :] = np.ones(
        (1, south_north_size, west_east_size), dtype=np.float32)

    f = ds.createVariable('F', 'f4', ('Time', 'south_north', 'west_east'))
    omega = 2 * np.pi / 86400
    f[:, :, :] = 2 * omega * np.sin(ds['XLAT'][:, :] * np.pi / 180)

    return wrf.getvar(ds, 'avo')


def calculate_geopotential(ds: Dataset):
    ph = wrf.getvar(ds, 'PH')
    phb = wrf.getvar(ds, 'PHB')
    hgt = ph + phb
    return wrf.destagger(hgt, 0, meta=True)


def calculate_relative_humidity(ds: Dataset):
    return wrf.getvar(ds, 'rh')


def calculate_pressure(ds: Dataset):
    return wrf.getvar(ds, 'p', units='hPa')


def calculate_temperature(ds: Dataset):
    return wrf.getvar(ds, 'T')


def calculate_uwind(ds: Dataset):
    return wrf.getvar(ds, 'ua', units='m s-1')


def calculate_vwind(ds: Dataset):
    return wrf.getvar(ds, 'va', units='m s-1')


def calculate_wwind(ds: Dataset):
    w = wrf.getvar(ds, 'wa', units='m s-1')

    # Convert to Pa/s
    p = calculate_pressure(ds)
    t = calculate_temperature(ds)
    rgas = 287.058
    rho = p / (rgas * t)
    omega = -w * rho * 9.80665
    omega.attrs['units'] = 'Pa s-1'

    return omega


def calculate_slp(ds: Dataset):
    return wrf.getvar(ds, 'slp', units='hPa')


def calculate_cape(ds: Dataset):
    return wrf.getvar(ds, 'cape2d')


def extract_at_levels(var: xr.DataArray, pressures: xr.DataArray, levels: 'list[int]'):
    return wrf.interplevel(var, pressures, levels)


def convert_longitudes_to_0_360(ds: Dataset):
    lon = ds['XLONG'][:, :, :]
    lon = np.where(lon < 0, lon + 360, lon)
    ds['XLONG'][:, :, :] = lon
    return ds


def extract_in_domain(ds: xr.Dataset, latrange: 'tuple[float, float]', lonrange: 'tuple[float, float]'):
    return ds.sel(dict(lat=slice(*latrange), lon=slice(*lonrange)))


def construct_xr_dataset(vars: 'dict[str, xr.DataArray]', lat: npt.ArrayLike, lon: npt.ArrayLike, lev: npt.ArrayLike, attrs: dict = None):
    def specify_dimensions_name(var: xr.DataArray):
        return ('lev', 'lat', 'lon') if var.ndim == 3 else ('lat', 'lon')

    def filter_attrs(attrs: dict):
        # Projection attribute cannot be serialized by xarray.
        if 'projection' in attrs:
            del attrs['projection']
        return attrs

    return xr.Dataset(
        {name: xr.Variable(specify_dimensions_name(var), var.data, filter_attrs(var.attrs)) for name, var in vars.items()},
        coords=dict(
            lat=lat,
            lon=lon,
            lev=lev,
        ),
        attrs=attrs,
    )


ExtractVariablesFnArgs = namedtuple(
    'ExtractVariablesFnArgs', ['path', 'outdir', 'prefix'])


def catch_extract_variables_exception(func: Callable[[ExtractVariablesFnArgs], None]):
    def wrapped_func(args: ExtractVariablesFnArgs):
        try:
            func(args)
        except Exception as e:
            logging.warning(f'=== IGNORE: {args.path} due to error:\n{e}')

    return wrapped_func


@catch_extract_variables_exception
def extract_variables(args: ExtractVariablesFnArgs):
    path = args.path
    outdir = args.outdir
    prefix = args.prefix

    assert os.path.isfile(path), f'Input file {path} not found.'

    # Load input file.
    ds = Dataset(path, 'a', diskless=True, persist=False)

    ds = convert_longitudes_to_0_360(ds)
    time = ds.variables['XTIME']
    time = num2date(time[:], time.units)
    assert len(time) == 1, 'Only works with data at a time.'

    pressures = calculate_pressure(ds)
    variables = [
        ['absvprs', calculate_vorticity, True],
        # ['capesfc', calculate_cape, False],
        ['hgtprs', calculate_geopotential, True],
        ['rhprs', calculate_relative_humidity, True],
        ['tmpprs', calculate_temperature, True],
        # tmpsfc is missing.
        ['ugrdprs', calculate_uwind, True],
        ['vgrdprs', calculate_vwind, True],
        ['vvelprs', calculate_wwind, True],
        ['slp', calculate_slp, False],
    ]

    # Extract each variable.
    v = OrderedDict()
    for name, fn, do_extract_at_levels in variables:
        try:
            values = fn(ds)
            if do_extract_at_levels:
                values = extract_at_levels(values, pressures, PRESSURE_LEVELS)

            v[name] = values
            logging.info(f'{name} is extracted with shape {values.data.shape}.')
        except (ValueError, KeyError) as e:
            logging.warning(f'{name} cannot be extracted.', exc_info=True)

    # Convert to xr.Dataset.
    lat_coord = ds['XLAT'][0, :, 0]
    lon_coord = ds['XLONG'][0, 0, :]
    ds = construct_xr_dataset(v, lat_coord, lon_coord, np.array(PRESSURE_LEVELS), ds.__dict__)

    # Then, extract data from the desired region.
    ds = extract_in_domain(ds, (5.0, 45.0), (100.0, 260.0))

    ds.to_netcdf(
        os.path.join(outdir, f'{prefix}_{time[0].strftime(TIME_FORMAT)}.nc'),
        mode='w',
        format='NETCDF4',
    )


def main(args=None):
    args = parse_arguments()

    # Create directory to store output datasets.
    os.makedirs(args.outputdir, exist_ok=True)

    inputfiles = args.inputfiles

    with Pool() as pool:
        tasks = pool.imap_unordered(
            extract_variables,
            [ExtractVariablesFnArgs(f, args.outputdir, args.prefix) for f in inputfiles])

        # Execute the tasks.
        for _ in tqdm(tasks, total=len(inputfiles)):
            pass


if __name__ == '__main__':
    main()
