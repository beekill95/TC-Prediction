from __future__ import annotations

import glob
import logging
import numpy as np
import os
from skimage import transform
import tensorflow as tf
import xarray as xr


from .utils import *


class PatchesDataLoader():
    """
    This data loader will load extracted dataset.
    """
    def __init__(self, subset: SubsetDict, output_size: tuple[int, int]) -> None:
        """
        Init the class.

        Parameters
        ==========
        domain_size: float
            Domain size of each square patch.
        """
        self._subset = subset
        self._output_size = output_size

    def load_dataset(self, path: str, batch_size: int) -> tf.data.Dataset:
        ds = list_nc_files(path)

        ds = ds.map(lambda path: tf.py_function(
                lambda p: load_xr_dataset(
                    p.numpy().decode(),
                    subset=self._subset,
                    output_size=self._output_size),
                [path],
                Tout=[tf.float64, tf.float64, tf.string],
                name='load_xr_dataset'),
             num_parallel_calls=tf.data.AUTOTUNE)

        # ds = ds.cache()

        ds = ds.batch(batch_size)
        print('new code')
        return ds.prefetch(tf.data.AUTOTUNE)


def list_nc_files(folder: str) -> tf.data.Dataset:
    files = glob.iglob(os.path.join(folder, '*.nc'))
    files = sorted(files)
    return tf.data.Dataset.from_tensor_slices(files)


def load_xr_dataset(path: str, *, subset: SubsetDict, output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        original_fn = extract_original_filename(path)
        ds = xr.load_dataset(path, engine='netcdf4')
        lat, lon = ds['lat'].values.min(), ds['lon'].values.min()
        ds = fill_missing_values(ds)
        ds = extract_subset(ds, subset)
        ds = transform.resize(ds, output_shape=output_size, preserve_range=True)
        return ds, np.asarray([lat, lon]), original_fn
    except Exception as e:
        logging.error(f'Cannot load dataset from {path=}')
        raise e


def extract_original_filename(path: str) -> str:
    filename, ext = os.path.splitext(os.path.basename(path))
    original_parts = filename.split('_')[:4]
    return '_'.join(original_parts) + ext
