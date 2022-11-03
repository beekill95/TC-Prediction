from __future__ import annotations

from .. import label
from .. import utils as data_utils

import abc
from datetime import datetime, timedelta
from functools import partial
import os
import pandas as pd
import tensorflow as tf


class TimeSeriesTropicalCycloneDataLoaderV2:
    def __init__(self, data_shape, previous_hours: list[int] = [6, 12, 18], subset=None):
        self._data_shape = data_shape
        self._previous_hours = previous_hours
        self._subset = subset

    @abc.abstractmethod
    def _process_to_dataset(self, tc_df: pd.DataFrame) -> tf.data.Dataset:
        pass

    def load_dataset(
            self,
            data_path,
            shuffle=False,
            batch_size=64,
            leadtimes: list[int]=None,
            non_tc_genesis_ratio=None,
            **kwargs):
        # Load TC dataframe.
        print('Dataframe loading.')
        tc_df = label.load_label(
                data_path,
                group_observation_by_date=True,
                leadtime=leadtimes)
        print('Dataframe in memory')
        tc_df['Path'] = tc_df['Path'].apply(
            partial(_add_previous_observation_data_paths,
                    previous_times=self._previous_hours))
        print('Add previous hours')
        tc_df = tc_df[tc_df['Path'].apply(_are_valid_paths)]
        print('Check previous hours valid')
        print(f'Remaining rows: {len(tc_df)}')

        # TODO:
        # can we move this into the dataset pipeline,
        # thus we can train the whole dataset without worrying about
        # unbalanced data.
        tc_df = data_utils.filter_negative_samples(
            tc_df,
            negative_samples_ratio=non_tc_genesis_ratio)

        print('Dataframe loaded')

        # Convert to tf dataset.
        dataset = self._process_to_dataset(tc_df, **kwargs)

        if shuffle:
            dataset = dataset.shuffle(batch_size * 3)

        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)

    @abc.abstractmethod
    def load_single_data(self, data_path):
        ...


def _add_previous_observation_data_paths(path: str, previous_times: list[int]) -> list[str]:
    dirpath = os.path.dirname(path)
    name = os.path.basename(path)
    name, _ = os.path.splitext(name)
    # The date of the observation is embedded in the filename: `fnl_%Y%m%d_%H_%M.nc`
    name_parts = name.split('_')
    name_prefix = name_parts[0]
    date_part = ''.join('_'.join(name_parts[1:]))

    date = datetime.strptime(date_part, '%Y%m%d_%H_%M')

    previous_times = previous_times.copy()
    previous_times.sort()
    dates = [date - timedelta(hours=time) for time in previous_times]
    dates += [date]
    return [os.path.join(dirpath, f"{name_prefix}_{d.strftime('%Y%m%d_%H_%M')}.nc") for d in dates]


def _are_valid_paths(paths: list[str]) -> bool:
    return all([os.path.isfile(p) for p in paths])
