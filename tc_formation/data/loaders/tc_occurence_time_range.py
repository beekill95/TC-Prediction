from .. import tfd_utils as tfd_utils
from .. import utils as data_utils
from ..time_series_addons import SingleTimeStepMixin
from .time_range import TimeSeriesTimeRangeDataLoader

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import xarray as xr


class TimeSeriesTropicalCycloneOccurenceTimeRangeDataLoader(TimeSeriesTimeRangeDataLoader):
    def _process_to_dataset(self, label_df: pd.DataFrame) -> tf.data.Dataset:
        time_ranges = len(label_df['Genesis'].iloc[0])


        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': np.asarray(label_df['Path']),
            'Genesis': np.asarray(label_df['Genesis']),
        })

        dataset = dataset.map(
            lambda row: tfd_utils.new_py_function(
                    lambda row: _load_observations(
                            [path.decode('utf-8') for path in row['Path'].numpy()],
                            row['Genesis'].numpy(),
                            self._subset,
                        ),
                    inp=[row],
                    Tout=[tf.float32, tf.float32],
                    name='load_observations',
                ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Tensorflow should figure out the shape of the output of previous map,
        # but it doesn't, so we have to do it ourself.
        # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
        dataset = dataset.map(
            lambda X, y: _set_dataset_shape(
                X, y,
                observation_shape=(len(self._previous_hours) + 1,) + self._data_shape,
                genesis_time_ranges=time_ranges,))

        return dataset

    def load_single_data(self, row) -> tuple[npt.NDArray, npt.NDArray]:
        return super().load_single_data(row)


class TropicalCycloneOccurenceTimeRangeDataLoader(SingleTimeStepMixin, TimeSeriesTropicalCycloneOccurenceTimeRangeDataLoader):
    pass


def _load_observations(paths: list[str], genesis: npt.NDArray, subset: dict = None) -> tuple[npt.NDArray, npt.NDArray]:
    observations = []
    for path in paths:
        dataset = xr.load_dataset(path)
        dataset = data_utils.extract_variables_from_dataset(dataset, subset)
        observations.append(np.expand_dims(dataset, axis=0))

    observations = np.concatenate(observations, axis=0)

    return observations, genesis


def _set_dataset_shape(
        observations: tf.Tensor, genesis: tf.Tensor,
        observation_shape: tuple[int, int, int, int], genesis_time_ranges: int):
    observations.set_shape(observation_shape)
    genesis.set_shape([genesis_time_ranges])
    return observations, genesis
