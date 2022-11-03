import abc
from ast import literal_eval
from datetime import datetime, timedelta
import numpy.typing as npt
import os
import pandas as pd
import tensorflow as tf


_TIME_STR_FMT = '%Y%m%d_%H_%M'

def load_time_range_label(path: str) -> pd.DataFrame:
    assert os.path.isfile(path), f'Invalid time range label path: {path}'
    df = pd.read_csv(
        path,
        converters=dict(
            Genesis=literal_eval,
            Genesis_Location=literal_eval,
            Genesis_SID=literal_eval,
            Other_TC=literal_eval,
        ))
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')

    return df



class TimeSeriesTimeRangeDataLoader(abc.ABC):
    def __init__(self,
            data_shape: tuple[int, int, int],
            previous_hours: list[int] = [],
            subset: dict = None) -> None:
        self._data_shape = data_shape
        self._previous_hours = previous_hours
        self._subset = subset

    @abc.abstractmethod
    def _process_to_dataset(self, label_df: pd.DataFrame) -> tf.data.Dataset:
        pass

    @abc.abstractmethod
    def load_single_data(self, row) -> tuple[npt.NDArray, npt.NDArray]:
        pass

    def load_dataset(
            self,
            path: str,
            batch_size: int = 128,
            shuffle: bool = False,
            caching: bool = True) -> tf.data.Dataset:

        label_df = load_time_range_label(path)
        print(label_df.columns)
        label_df['Path'] = label_df.apply(
                lambda row: self._add_previous_hours_paths(row['Date'], row['Path']),
                axis=1)
        label_df = label_df[label_df['Path'].apply(_are_all_paths_valid)]
        dataset = self._process_to_dataset(label_df)

        if caching:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(batch_size * 3)

        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)


    def _add_previous_hours_paths(self, current: datetime, path: str) -> list[str]:
        parent_dir = os.path.dirname(path)
        filename, _ = os.path.splitext(os.path.basename(path))
        prefix = filename.split('_')[0]

        paths = [path]

        for previous_hour in self._previous_hours:
            previous_date = current - timedelta(hours=previous_hour)
            previous_date_fn = f'{prefix}_{previous_date.strftime(_TIME_STR_FMT)}.nc'

            previous_date_path = os.path.join(parent_dir, previous_date_fn)
            paths.append(previous_date_path)

        return paths


def _are_all_paths_valid(paths: list[str]) -> bool:
    return all(os.path.isfile(path) for path in paths)
