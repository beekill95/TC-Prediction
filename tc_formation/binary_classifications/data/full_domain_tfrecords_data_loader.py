from __future__ import annotations

import numpy as np
import tensorflow as tf

from ...data.tfd_utils import new_py_function


class FullDomainTFRecordsDataLoader():
    def __init__(self, datashape: tuple[int, ...]):
        self._datashape = datashape

    def load_dataset(self, path: str) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(_parse_tfrecords)
        ds = ds.map(
            lambda d: new_py_function(
                lambda d: _parse_binary_dataset(
                    data=d['data'],
                    datashape=d['data_shape'],
                    genesis_locations=d['genesis_locations'],
                    genesis_locations_shape=d['genesis_locations_shape'],
                    filename=d['filename'],
                    genesis_date=d['genesis_date'],
                    file_date=d['file_date'],
                ),
                [d],
                Tout=[tf.float32, tf.float32, tf.string, tf.float32, tf.float32],
                name='parse_binary_dataset',
            ),
            num_parallel_calls=tf.data.AUTOTUNE)
        # Tensorflow doesn't know the shape of the output data :(((
        ds = ds.map(_set_data_shape(self._datashape))
        ds = ds.cache()
        
        return ds.prefetch(tf.data.AUTOTUNE)


_patches_dataset_description = dict(
    data=tf.io.FixedLenFeature([], tf.string),
    data_shape=tf.io.RaggedFeature(dtype=tf.int64),
    genesis_locations=tf.io.FixedLenFeature([], tf.string),
    genesis_locations_shape=tf.io.RaggedFeature(dtype=tf.int64),
    filename=tf.io.FixedLenFeature([], tf.string),
    genesis_date=tf.io.RaggedFeature(dtype=tf.float32),
    file_date=tf.io.RaggedFeature(dtype=tf.float32),
)

def _parse_tfrecords(example_proto):
    results = tf.io.parse_single_example(
        example_proto, _patches_dataset_description)
    return results


def _parse_binary_dataset(*, data, datashape, genesis_locations, genesis_locations_shape, filename, genesis_date, file_date):
    data = np.frombuffer(data.numpy(), dtype=np.float32).reshape(datashape.numpy())
    genesis_locations = np.frombuffer(genesis_locations.numpy(), dtype=np.float32).reshape(genesis_locations_shape.numpy())
    return data, genesis_locations, filename.numpy().decode('utf-8'), genesis_date, file_date


def _set_data_shape(datashape):
    def _set_shape(data, *args):
        data.set_shape(datashape)
        return data, *args

    return _set_shape
