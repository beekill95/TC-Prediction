from __future__ import annotations

import numpy as np
import tensorflow as tf
import logging

from .full_domain_tfrecords_data_loader import FullDomainTFRecordsDataLoader


logger = logging.getLogger(__name__)


class RandomPositivePatchesDataLoader(FullDomainTFRecordsDataLoader):
    def __init__(self, datashape: tuple[int, ...], domain_size: int, margin: int = 5):
        super().__init__(datashape)

        self._domain_size = domain_size
        self._margin = margin

    def load_dataset(self, path: str) -> tf.data.Dataset:
        ds = super().load_dataset(path)
        ds = ds.map(
            self.autocrop_and_label,
            num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(self.set_shape)

        return ds

    def autocrop_and_label(self, d, locations, *_):
        return tf.numpy_function(
            lambda data, locations:
                autocrop_around_genesis_locations(data, locations, self._domain_size, self._margin),
            inp=[d, locations],
            Tout=[tf.float64, tf.int64],
            name='autocrop_and_label')

    def set_shape(self, X, y):
        nb_channels = self._datashape[-1]
        X.set_shape((self._domain_size, self._domain_size, nb_channels))
        y.set_shape([1])
        return X, y


def autocrop_around_genesis_locations(data: np.ndarray, genesis_locations: np.ndarray, domain_size: int, margin: int):
    # First, randomly choose one genesis location for generating positive patch.
    nb_genesis = genesis_locations.shape[0]
    if nb_genesis == 1:
        loc = genesis_locations[0]
    else:
        loc = genesis_locations[np.random.choice(nb_genesis)]
    # print(genesis_locations, 'chosen location', loc)

    # The `loc` is in pixel coordinate.
    # Now, we will have to generate a valid domain for randomization.
    data_shape = data.shape
    valid_x = find_valid_pixel_range(loc[0], lower=0, upper=data_shape[0], size=domain_size, margin=margin)
    valid_y = find_valid_pixel_range(loc[1], lower=0, upper=data_shape[1], size=domain_size, margin=margin)

    # Finally, choose a random x and y.
    try:
        x, y = tuple(np.random.choice(r) for r in (valid_x, valid_y))
        return data[x:x+domain_size, y:y+domain_size].astype(np.float64), [1]
    except ValueError:
        logger.warning('Couldnt find a valid random positive patch, so a default negative patch is returned instead.')
        return data[:domain_size, :domain_size].astype(np.float64), [0]


def find_valid_pixel_range(loc: int, *, lower: int, upper: int, size: int, margin: int = 5) -> np.ndarray:
    """
    This function will find a valid pixel range,
    such that when choosing a random pixel x within this range,
    it ensures that:
    * x + slack <= loc <= x + size - slack
    * x >= lower
    * x + size <= upper
    """
    all_pixels = np.arange(lower, upper)
    upper_constraints = (all_pixels <= loc - margin) & (all_pixels <= upper - size)
    lower_constraints = (all_pixels >= lower) & (all_pixels >= loc - size + margin)
    results = all_pixels[lower_constraints & upper_constraints]
    # If we cannot find the results that satisfy the `margin`,
    # try again with margin = 0.
    if len(results) == 0 and not margin == 0:
        return find_valid_pixel_range(loc, lower=lower, upper=upper, size=size, margin=0)

    # print(f'{loc=}, {upper=}, {size=}, {slack=}\n{results=}')
    return results
