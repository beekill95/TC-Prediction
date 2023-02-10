from datetime import datetime
import numpy as np
import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def numpy_feature(value: np.ndarray, dtype=np.float32):
    value_bytes = value.astype(dtype).tobytes()
    return bytes_feature(value_bytes)


def date_feature(date: datetime):
    datenum = date.timestamp()
    return float_feature([datenum])
