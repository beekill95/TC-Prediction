import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xarray as xr


def _extract_date_from_observation_path(path):
    filename = Path(path).stem
    return ''.join(filename.split('_')[1:-1])


def load_data(data_dir, batch_size=32, shuffle=False, negative_samples_ratio=None):
    """
    Load data from the given directory.

    :param data_dir: path to directory contains observation data *.nc and label file tc.csv
    :param batch_size: how many observation data should be in a batch.
    :param shuffle: should shuffle data after we have exhausted data points.
    :param negative_samples_ratio: (default: None) the ratio of negative samples to positive samples.
    For instance, if the ratio is 3, then for 1 positive sample, there will be three negative samples.
    If None is passed, all the negative samples are taken.
    :returns:
    """
    # Get a list of all observation data files.
    observations = glob.glob(os.path.join(data_dir, '*.nc'))
    observations = pd.DataFrame(observations, columns=['Path'])
    observations['Date'] = observations['Path'].apply(
        _extract_date_from_observation_path)

    # Load labels.
    labels = pd.read_csv(
        os.path.join(data_dir, 'tc.csv'),
        dtype={
            'Observation': str,
            'TC': np.int,
            'Genesis': str,
            'End': str,
            'Latitude': str,
            'Longitude': str,
        })

    # Merge observations and labels into single dataframe.
    dataset = pd.merge(
        observations,
        labels,
        how='inner',
        left_on='Date',
        right_on='Observation')
    dataset = _filter_negative_samples(dataset, negative_samples_ratio)

    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset['Path'], dataset['TC']))
    if shuffle:
        dataset = dataset.shuffle(len(observations))

    # Load given dataset to memory.
    dataset = dataset.map(lambda path, tc: tf.numpy_function(
        _load_observation_data,
        inp=[path, tc],
        Tout=[tf.float32, tf.int64],
        name='load_observation_data'),
        num_parallel_calls=2,
        deterministic=False,
    )

    # Tensorflow should figure out the shape of the output of previous map,
    # but it doesn't, so we have to do it our self.
    # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
    dataset = dataset.map(lambda observation, tc: _set_shape(observation, tc))

    return dataset.batch(batch_size)


def _load_observation_data(observation_path, label):
    data = xr.open_dataset(observation_path.decode('utf-8'), engine='netcdf4')
    data = data.to_array().values

    # Reshape data so that it have channel_last format.
    data = np.moveaxis(data, 0, -1)

    return data, [label]


def _set_shape(observation, label):
    # TODO: we shouldn't fixed shape here!!!!
    observation.set_shape([41, 181, 5])
    label.set_shape([1])
    return observation, label


def _filter_negative_samples(dataset, negative_samples_ratio):
    if negative_samples_ratio is None:
        return dataset

    positive_samples = dataset[dataset['TC'] == 1]
    negative_samples = dataset[dataset['TC'] == 0]

    nb_negative_samples_to_take = int(
        len(positive_samples) * negative_samples_ratio)
    negative_samples = negative_samples.sample(nb_negative_samples_to_take)

    result = pd.concat([positive_samples, negative_samples])
    return result.sort_values(by='Observation').reset_index()


if __name__ == '__main__':
    a = load_data(
        '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_test/6h_700mb',
        batch_size=1,
        negative_samples_ratio=3)
    a = iter(a)
    for i in range(1000):
        # print(i)
        b = next(a)
    print(b)
