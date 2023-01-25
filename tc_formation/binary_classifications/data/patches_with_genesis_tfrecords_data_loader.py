from __future__ import annotations

import numpy as np
import tensorflow as tf

from ...data.tfd_utils import new_py_function


class PatchesWithGenesisTFRecordDataLoader():
    def load_dataset(self, path: str, batch_size: int) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(_parse_dataset)
        ds = ds.map(
            lambda d: new_py_function(
                lambda d: _parse_binary_dataset(
                    data=d['data'],
                    datashape=d['data_shape'],
                    position=d['position'],
                    filename=d['filename'],
                    genesis=d['genesis'],
                ),
                [d],
                Tout=[tf.float64, tf.float64, tf.string, tf.int64],
                name='parse_binary_dataset',
            ),
            num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda d, _a, _b, g: (d, g)).cache()

        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


_patches_dataset_description = dict(
    data=tf.io.FixedLenFeature([], tf.string),
    data_shape=tf.io.RaggedFeature(dtype=tf.int64),
    genesis=tf.io.RaggedFeature(dtype=tf.int64),
    position=tf.io.FixedLenFeature([], tf.string),
    filename=tf.io.FixedLenFeature([], tf.string),
)

def _parse_dataset(example_proto):
    results = tf.io.parse_single_example(
        example_proto, _patches_dataset_description)
    return results


def _parse_binary_dataset(*, data, datashape, position, filename, genesis):
    data = np.frombuffer(data.numpy(), dtype=np.float32).reshape(datashape.numpy())
    position = np.frombuffer(position.numpy(), dtype=np.float32)
    return data, position, filename.numpy().decode('utf-8'), genesis[0]
