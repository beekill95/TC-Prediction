from __future__ import annotations

from .time_series_v2 import TimeSeriesTropicalCycloneDataLoaderV2
from .. import tfd_utils as tfd_utils

import numpy as np
import pandas as pd
import tensorflow as tf


class TimeSeriesTCBinaryClassificationLoader():
    def __init__(self, domain_size: int, data_shape: tuple[int, int, int], previous_hours: list[int], subset) -> None:
        self._positive_tc_loader = _TimeSeriesPositiveTCLoader(
                domain_size, data_shape=data_shape, previous_hours=previous_hours, subset=subset)

    def load_dataset(self, data_path, shuffle=False, batch_size=64, leadtimes: list[int] = None, **kwargs):
        pass


class _TimeSeriesPositiveTCLoader(TimeSeriesTropicalCycloneDataLoaderV2):
    def __init__(self, domain_size: int, data_shape: tuple[int, int, int], previous_hours: list[int], subset):
        super().__init__(data_shape, previous_hours=previous_hours, subset=subset)
        self._domain_size = domain_size

    def load_dataset(self,
            data_path, 
            shuffle=False,
            batch_size=64,
            leadtimes:list[int]=None,
            **kwargs):
        return super().load_dataset(
            data_path,
            shuffle=shuffle,
            batch_size=batch_size,
            leadtimes=leadtimes,
            non_tc_genesis_ratio=0,
            **kwargs)

    def _process_to_dataset(self, label_df: pd.DataFrame) -> tf.data.Dataset:
        # `tc_df` contains only genesis cases.
        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': label_df['Path'].tolist(),
            'Latitude': label_df['Latitude'].values,
            'Longitude': label_df['Longitude'].values,
        })

        dataset = dataset.map(
            lambda row: tfd_utils.new_py_function(
                lambda row: _load_reanalysis(
                    [path.decode('utf-8') for path in row['Path'].numpy()],
                    row['Latitude'],
                    row['Longitude'],
                    self._subset),
                inp=[row],
                Tout=[tf.float32, tf.float32],
                name='load_reanalysis',
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # So we don't have to load data again from disk.
        dataset = dataset.cache()

        # Extract a patch that contains TC.
        dataset = dataset.map(
            lambda obs, loc: _extract_tc_patch(obs, loc, self._domain_size))

        # Include target.
        dataset = dataset.map(lambda obs: obs, [True])

        return dataset


def _load_reanalysis(paths: list[str], latitude: float, longitude: float, subset: dict):
    pass


def _extract_tc_patch(observations, loc: tuple[float, float], domain_size: int):
    """
    Assume that the observations' resolution is 1 degree.
    `domain_size` is also in degree.
    """
    pass
