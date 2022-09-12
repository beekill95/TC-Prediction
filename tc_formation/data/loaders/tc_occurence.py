from .. import label as label
from .. import tfd_utils as tfd_utils
from .. import utils as data_utils
from ..time_series import TimeSeriesTropicalCycloneDataLoader
from ..time_series_addons import SingleTimeStepMixin

from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple
import xarray as xr



class TimeSeriesTropicalCycloneOccurenceDataLoader(TimeSeriesTropicalCycloneDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_to_dataset(self, tc_df: pd.DataFrame) -> tf.data.Dataset:
        cls = TimeSeriesTropicalCycloneOccurenceDataLoader

        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': np.asarray(tc_df['Path'].sum()).reshape((-1, len(self._previous_hours) + 1)),
            'TC': tc_df['TC'],
        })
        
        dataset = dataset.map(
            lambda row: tfd_utils.new_py_function(
                    lambda row: cls._load_reanalysis(
                            [path.decode('utf-8') for path in row['Path'].numpy()],
                            self._subset,
                            row['TC'].numpy(),
                        ),
                    inp=[row],
                    Tout=[tf.float32, tf.float32],
                    name='load_observation_and_gt',
                ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Tensorflow should figure out the shape of the output of previous map,
        # but it doesn't, so we have to do it ourself.
        # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
        dataset = dataset.map(partial(
            cls._set_dataset_shape,
            shape=(len(self._previous_hours) + 1,) + self._data_shape,
        ))
        
        return dataset

    @classmethod
    def _load_reanalysis(
            cls,
            paths: List[str],
            subset: dict,
            has_tc: bool,
        ) -> Tuple[np.ndarray, np.ndarray]:

        datasets = []
        for path in paths:
            dataset = xr.open_dataset(path, engine='netcdf4')
            dataset = data_utils.extract_variables_from_dataset(dataset, subset)
            datasets.append(np.expand_dims(dataset, axis=0))
        datasets = np.concatenate(datasets, axis=0)

        return datasets, np.array([has_tc])

    @classmethod
    def _set_dataset_shape(
            cls,
            data: tf.Tensor,
            has_tc: tf.Tensor,
            shape: tuple):
        data.set_shape(shape)
        has_tc.set_shape((1,))
        return data, has_tc


class TropicalCycloneOccurenceDataLoader(SingleTimeStepMixin, TimeSeriesTropicalCycloneOccurenceDataLoader):
    ...
