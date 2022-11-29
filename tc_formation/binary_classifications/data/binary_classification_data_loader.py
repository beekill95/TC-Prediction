from __future__ import annotations

from collections import OrderedDict
import glob
import json
import numpy as np
import os
from skimage import transform
import tensorflow as tf
from typing import Union
import xarray as xr

SubsetDict = OrderedDict[str, Union[tuple[float, ...], bool]]

class BinaryClassificationDataLoader():
    _cache_version = '1.0'

    def __init__(self, resize_shape: tuple[int, int], sel: SubsetDict):
        self._subset = self._normalize_subset_dict(sel)
        self._cache_dir = self._generate_cache_parent_dir(self._subset)
        self._resize_shape = resize_shape

    def load_dataset(
            self, data_dir: str, batch_size=64, shuffle=True,
            val_split=0.0, test_split=0.0) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        def split_train_val(patches: tf.data.Dataset):
            assert (val_split + test_split) <= 1.0

            nb_patches = patches.cardinality().numpy()

            train_size = int(nb_patches * (1 - val_split - test_split))
            val_size = int(nb_patches * val_split)
            test_size = nb_patches - train_size - val_size

            train_patches = patches.take(train_size)
            val_patches = patches.skip(train_size).take(val_size)
            test_patches = patches.skip(train_size + val_size).take(test_size)
            return train_patches, val_patches, test_patches

        # Make sure that the cache dir exist.
        os.makedirs(self._cache_dir, exist_ok=True)

        pos_dir = os.path.join(data_dir, 'pos')
        pos_patches = load_dataset_with_label(pos_dir, 1, self._subset, self._resize_shape)
        # pos_patches = load_dataset_with_label(pos_dir, 1, self._subset, self._resize_shape)

        neg_dir = os.path.join(data_dir, 'neg')
        neg_patches = load_dataset_with_label(neg_dir, 0, self._subset, self._resize_shape)
        # neg_patches = load_dataset_with_label(neg_dir, 0, self._subset, self._resize_shape)

        train_pos_patches, val_pos_patches, test_pos_patches = split_train_val(pos_patches)
        train_neg_patches, val_neg_patches, test_neg_patches = split_train_val(neg_patches)

        train_patches = train_pos_patches.concatenate(train_neg_patches)
        val_patches = val_pos_patches.concatenate(val_neg_patches)
        test_patches = test_pos_patches.concatenate(test_neg_patches)

        train_patches = train_patches.cache()
        val_patches = val_patches.cache()
        test_patches = test_patches.cache()
        
        if shuffle:
            train_patches = train_patches.shuffle(batch_size * 3)

        train_patches = train_patches.batch(batch_size)
        val_patches = val_patches.batch(batch_size)
        test_patches = test_patches.batch(batch_size)

        return train_patches.prefetch(2), val_patches.prefetch(2), test_patches.prefetch(2)

    def _generate_cache_parent_dir(self, subset: SubsetDict) -> str:
        json_str = json.dumps(subset)
        hashed = str(hash(json_str))
        return os.path.join(
            'data/.cache/binary_classification',
            self._cache_version, hashed)

    def _normalize_subset_dict(self, subset: SubsetDict) -> SubsetDict:
        d = OrderedDict()
        for key, values in subset.items():
            if not(isinstance(values, bool) and not values):
                d[key.lower()] = values
        
        return d


def list_nc_files(folder: str) -> tf.data.Dataset:
    files = glob.glob(os.path.join(folder, '*.nc'))
    files = sorted(files)
    return tf.data.Dataset.from_tensor_slices(files)


def fill_nan_with_mean(values: np.ndarray) -> np.ndarray:
    means = np.nanmean(values, axis=(1, 2), keepdims=True)
    nanmask = np.isnan(values)
    return np.where(nanmask, means * np.ones_like(values), values)


def load_xr_dataset_as_numpy_array(
        path: str, subset: SubsetDict, output_size: tuple[int, int]): # -> dict[str, tf.Tensor]:
    ds = xr.load_dataset(path, engine='netcdf4')
    tensors = []
    for key, lev in subset.items():
        values = None
        if isinstance(lev, bool):
            if lev:
                values = ds[key].values
        else:
            try:
                values = ds[key].sel(lev=list(lev)).values
            except Exception:
                print('Error', path, ds[key]['lev'])
                raise ValueError('error')

        if values is not None:
            if values.ndim == 2:
                values = values[None, ...]

            values = fill_nan_with_mean(values)
            tensors.append(values)

    tensors = np.concatenate(tensors, axis=0)
    tensors = np.moveaxis(tensors, 0, -1)
    tensors = transform.resize(tensors, output_size, preserve_range=True)
    tensors = tf.convert_to_tensor(tensors, dtype=tf.float64)

    return tensors


def load_dataset_with_label(
        data_dir: str,
        label: int,
        subset: SubsetDict,
        output_size: tuple[int, int]) -> tf.data.Dataset:
    patches = (list_nc_files(data_dir)
        .map(lambda path: tf.py_function(
                lambda p: load_xr_dataset_as_numpy_array(
                    p.numpy().decode(), subset, output_size),
                [path],
                Tout=tf.float64,
                name='load_xr_dataset'),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda X: (X, tf.constant(label, dtype=tf.int64))))
    return patches
