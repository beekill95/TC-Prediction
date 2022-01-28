import tensorflow as tf
from tc_formation.data.time_series import TimeSeriesTropicalCycloneDataLoader, TimeSeriesTropicalCycloneWithGridProbabilityDataLoader
from typing import List


class TimeSeriesDataLoaderAddon(TimeSeriesTropicalCycloneDataLoader):
    def __init__(self, data_loader: TimeSeriesTropicalCycloneDataLoader):
        self._data_loader = data_loader


class WithPriorTCProbAddon(TimeSeriesDataLoaderAddon):
    def __init__(self, data_loader: TimeSeriesTropicalCycloneWithGridProbabilityDataLoader, grid_prob_output_name: str, tc_prob_prior_output_name: str):
        super().__init__(data_loader)
        self._grid_prob_output_name = grid_prob_output_name
        self._tc_prob_prior_output_name = tc_prob_prior_output_name

    def load_dataset(self, *args, **kwargs):
        cls = WithPriorTCProbAddon

        dataset = self._data_loader.load_dataset(*args, **kwargs)

        return dataset.map(lambda X, y: cls.attach_prior_prob(
            X, y,
            self._data_loader._softmax_output,
            self._grid_prob_output_name,
            self._tc_prob_prior_output_name))

    @classmethod
    def attach_prior_prob(cls, X, y, softmax_output: bool, grid_prob_name: str, tc_prob_prior_name: str):
        y_ = y[:, :, :, 1] if softmax_output else y
        prior_prob = tf.reduce_any(y_ > 0, axis=[1, 2])
        prior_prob = tf.cast(prior_prob, dtype=tf.float32)

        return X, {grid_prob_name: y, tc_prob_prior_name: prior_prob}
