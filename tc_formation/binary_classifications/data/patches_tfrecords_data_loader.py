from __future__ import annotations

import tensorflow as tf


class PatchesTFRecordDataLoader():
    def load_dataset(self, path: str, batch_size: int) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(_parse_dataset)

        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


_patches_dataset_description = dict(
    data=tf.io.FixedLenFeature([], tf.string),
    position=tf.io.FixedLenFeature([], tf.string),
    filename=tf.io.FixedLenFeature([], tf.string),
)

def _parse_dataset(example_proto):
    results = tf.io.parse_single_example(
        example_proto, _patches_dataset_description)
    return (results['data'].numpy(),
            results['position'].numpy(),
            results['filename'].numpy())

