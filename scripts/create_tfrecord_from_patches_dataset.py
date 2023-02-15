#!/usr/bin/env python3

from __future__ import annotations

try:
    from .tc_binary_classification_helpers import *
except ImportError:
    from tc_binary_classification_helpers import *

try:
    from .tfrecords_utils import *
except ImportError:
    from tfrecords_utils import *

import argparse
from collections import OrderedDict, namedtuple
import glob
from multiprocessing import Pool
import numpy as np
import os
from skimage import transform
import tensorflow as tf
from tqdm.auto import tqdm
import xarray as xr


SUBSET = OrderedDict(
    absvprs=(900, 750),
    rhprs=(750,),
    tmpprs=(900, 500),
    hgtprs=(500,),
    vvelprs=(500,),
    ugrdprs=(800, 200),
    vgrdprs=(800, 200),
    tmpsfc=True,
    pressfc=True,
)
VARIABLES_ORDER_FUTURE = [
    'absvprs',
    'hgtprs',
    'pressfc',
    'rhprs',
    'tmpprs',
    'tmpsfc',
    'ugrdprs',
    'vgrdprs',
    'vvelprs',
]


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('indir', help='Path to input directory.')
    parser.add_argument('filename', help='Output filename.')
    parser.add_argument(
        '--processes',
        type=int,
        default=8,
        help='Number of parallel processes. Default to 8.')
    parser.add_argument(
        '--output-size',
        dest='output_size',
        type=int,
        nargs=2,
        default=[30, 30],
        help='Size of output patch.',
    )
    parser.add_argument(
        '--all-variables',
        dest='all_variables',
        action='store_true',
        help='Whether should we extract all variables or not. Default is False, which means only a subset is extracted.'
    )

    return parser.parse_args(args)


def _to_example(value: np.ndarray, pos: np.ndarray, path: str):
    feature = dict(
        data=numpy_feature(value),
        data_shape=int64_feature(value.shape),
        position=numpy_feature(pos),
        filename=bytes_feature(str.encode(path)),
    )
    return tf.train.Example(features=tf.train.Features(feature=feature))


ProcessArgs = namedtuple('ProcessArgs', ['path', 'output_size', 'all_variables'])
def process_to_tf_example(args: ProcessArgs):
    path, output_size, all_variables = args.path, args.output_size, args.all_variables

    original_fn = extract_original_filename(path)
    ds = xr.load_dataset(path, engine='netcdf4')
    lat, lon = ds['lat'].values.min(), ds['lon'].values.min()
    ds = fill_missing_values(ds)
    ds = (extract_subset(ds, SUBSET)
             if not all_variables
             else extract_all_variables(ds, VARIABLES_ORDER_FUTURE))
    ds = transform.resize(ds, output_shape=output_size, preserve_range=True)
    example = _to_example(ds, np.asarray([lat, lon]), original_fn)
    return example.SerializeToString()


def extract_original_filename(path: str) -> str:
    filename, ext = os.path.splitext(os.path.basename(path))
    original_parts = filename.split('_')[:4]
    return '_'.join(original_parts) + ext


# def extract_subset(ds: xr.Dataset, subset: OrderedDict) -> np.ndarray:
#     tensors = []
#     for key, lev in subset.items():
#         values = None
#         if isinstance(lev, bool):
#             if lev:
#                 values = ds[key].values
#         else:
#             try:
#                 values = ds[key].sel(lev=list(lev)).values
#             except Exception as e:
#                 print(key, lev, ds[key]['lev'])
#                 raise e

#         if values is not None:
#             if values.ndim == 2:
#                 values = values[None, ...]

#             tensors.append(values)

#     tensors = np.concatenate(tensors, axis=0)
#     tensors = np.moveaxis(tensors, 0, -1)
#     return tensors


def fill_missing_values(ds: xr.Dataset) -> xr.Dataset:
    mean_values = ds.mean(dim=['lat', 'lon'], skipna=True)
    return ds.fillna(mean_values)


def main(args=None):
    args = parse_arguments(args)
    outpath = os.path.join(args.indir, f'{args.filename}.tfrecords')
    assert not os.path.isfile(outpath), f'File {outpath} exists!'

    files = glob.glob(os.path.join(args.indir, '*.nc'))

    with Pool(args.processes) as pool:
        tasks = pool.imap_unordered(
            process_to_tf_example, 
            (ProcessArgs(f, args.output_size, args.all_variables) for f in files))

        with tf.io.TFRecordWriter(outpath) as writer:
            for result in tqdm(tasks, total=len(files)):
                writer.write(result)


if __name__ == '__main__':
    main()
