from __future__ import annotations

from collections import OrderedDict
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
from skimage import transform
import tensorflow as tf
from typing import Union, Iterator
import xarray as xr


SubsetDict = OrderedDict[str, Union[tuple[float, ...], bool]]
KEEP_HOURS = [0, 6, 12, 18]


class PatchesClassificationDataLoader():
    """
    This data loader will load full observations as smaller patches.
    """
    def __init__(self, *, domain_size: float, stride: float, subset: SubsetDict, keep_hours: list[int] = KEEP_HOURS) -> None:
        """
        Init the class.

        Parameters
        ==========
        domain_size: float
            Domain size of each square patch.
        """
        self._domain_size = domain_size
        self._stride = stride
        self._subset = subset
        self._keep_hours = keep_hours

    def load_dataset_without_label(self, dirpath: str, batch_size: int = 64) -> tf.data.Dataset:
        """
        Load dataset without relying on typical csv file.

        Parameters
        ==========
        dirpath: str
            Path to directory containing .nc files.
        """
        ds = list_nc_files(dirpath, self._keep_hours)

        ds = ds.map(lambda path: tf.py_function(
                lambda p: load_xr_dataset_as_patches(
                    p.numpy().decode(),
                    subset=self._subset,
                    domain_size=self._domain_size,
                    stride=self._stride),
                [path],
                Tout=[tf.float64, tf.float64],
                name='load_xr_dataset_as_patches'),
             num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.cache()
        ds = ds.batch(batch_size)

        return ds.prefetch(1)

def list_nc_files(folder: str, keep_hours: list[int]) -> tf.data.Dataset:
    def is_time_of_day_valid(path: str) -> bool:
        filename, _ = os.path.splitext(os.path.basename(path))
        # filename has format: <prefix>_YYYYmmdd_HH_MM
        time_part = filename.split('_')[2]
        return int(time_part) in keep_hours

    files = glob.iglob(os.path.join(folder, '*.nc'))
    files = filter(is_time_of_day_valid, files)
    files = sorted(files)
    return tf.data.Dataset.from_tensor_slices(files)


def load_xr_dataset_as_patches(path: str, subset: SubsetDict, domain_size: float, stride: float) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.load_dataset(path, engine='netcdf4')
    ds = fill_missing_values(ds)

    try:
        patches, coords = zip(*extract_patches(ds, domain_size=domain_size, stride=stride))
        patches = (extract_subset(p, subset) for p in patches)
        patches = [p[None, ...] for p in patches]
        patches = resize_to_the_smallest_size(patches)
        patches = np.concatenate(patches, axis=0)
    except Exception as e:
        logging.error(f'Cannot extract patches from file: {path}', e)
        raise e

    return patches, np.asarray(coords)


def extract_patches(ds: xr.Dataset, domain_size: float, stride: float) -> Iterator[tuple[xr.Dataset, tuple[float, float]]]:
    """
    Extract smaller patches from the given dataset.
    By default, it will extract valid patches only.
    """
    lat, lon = ds['lat'].values, ds['lon'].values
    minlat, maxlat = lat.min(), lat.max()
    minlon, maxlon = lon.min(), lon.max()

    for domain_lower_lat in np.arange(minlat, maxlat, stride):
        domain_upper_lat = domain_lower_lat + domain_size

        for domain_lower_lon in np.arange(minlon, maxlon, stride):
            domain_upper_lon = domain_lower_lon + domain_size

            if (domain_upper_lat <= maxlat) and (domain_upper_lon <= maxlon):
                patch = ds.sel(
                    lat=slice(domain_lower_lat, domain_upper_lat),
                    lon=slice(domain_lower_lon, domain_upper_lon))
                yield patch, (domain_lower_lat, domain_lower_lon)


def extract_subset(ds: xr.Dataset, subset: SubsetDict) -> np.ndarray:
    tensors = []
    for key, lev in subset.items():
        values = None
        if isinstance(lev, bool):
            if lev:
                values = ds[key].values
        else:
            values = ds[key].sel(lev=list(lev)).values

        if values is not None:
            if values.ndim == 2:
                values = values[None, ...]

            tensors.append(values)

    tensors = np.concatenate(tensors, axis=0)
    tensors = np.moveaxis(tensors, 0, -1)
    return tensors


def resize_to_the_smallest_size(patches: list[np.ndarray]) -> list[np.ndarray]:
    # Find the smallest size.
    sizes = [p.shape for p in patches]
    smallest_size = min(sizes, key=lambda x: np.prod(x))

    # Resize to the smallest size.
    patches = [transform.resize(
        p, output_shape=smallest_size, preserve_range=True) for p in patches]
    return patches


def fill_missing_values(ds: xr.Dataset) -> xr.Dataset:
    mean_values = ds.mean(dim=['lat', 'lon'], skipna=True)
    return ds.fillna(mean_values)
