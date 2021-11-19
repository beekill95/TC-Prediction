import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xarray as xr
from typing import Union, List
from datetime import timedelta
from functools import reduce


def _extract_date_from_observation_path(path):
    filename = Path(path).stem
    return ''.join(filename.split('_')[1:-1])

def parse_tc_datetime(column: pd.Series):
    return pd.to_datetime(column, format='%Y-%m-%d %H:%M:%S')


def load_tc_with_observation_path(data_dir):
    """
    Load tc.csv and all available observation files into one nice dataframe for easy processing.

    :param data_dir: path to directory contains all observation files and tc.csv
    :returns: a dataframe contains the content of tc.csv and path to corresponding observation file.
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
            'TC': int,
            'Genesis': str,
            'End': str,
            'Latitude': str,
            'Longitude': str,
        })

    # Merge observations and labels into single dataframe,
    # and return the result.
    return pd.merge(
        observations,
        labels,
        how='inner',
        left_on='Date',
        right_on='Observation')


def extract_variables_from_dataset(dataset: xr.Dataset, subset: dict = None):
    data = []
    for var in dataset.data_vars:
        if subset is not None and var in subset:
            values = dataset[var].sel(lev=subset[var]).values
        else:
            values = dataset[var].values

        # For 2D dataarray, make it 3D.
        if len(np.shape(values)) != 3:
            values = np.expand_dims(values, 0)

        data.append(values)

    # Reshape data so that it have channel_last format.
    data = np.concatenate(data, axis=0)
    data = np.moveaxis(data, 0, -1)

    return data

def filter_in_leadtime(tc: pd.DataFrame, leadtimes: Union[List[int], int] = None):
    if leadtimes is None:
        return tc

    if not isinstance(leadtimes, list):
        leadtimes = [leadtimes]

    # First, we will keep all negative cases.
    mask = ~tc['TC']

    # Then, loop through each lead time to get observations that belong to that leadtime.
    observation_dates = parse_tc_datetime(tc['Observation'])
    tc_first_observed_dates = parse_tc_datetime(tc['First Observed'])
    for leadtime in leadtimes:
        leadtime = timedelta(hours=leadtime)
        mask |= (tc_first_observed_dates - observation_dates) == leadtime

    return tc[mask]

def group_observations_by_date(tc_labels: pd.DataFrame):
    def concat_values(values):
        return reduce(lambda agg, x: agg + [x], values, [])

    grouped = tc_labels.groupby('Date')

    tc_labels['TC'] = grouped['TC'].transform(
            lambda has_tc: reduce(lambda agg, x: agg and x, has_tc, True))

    concate_columns = [
            'TC Id',
            'First Observed',
            'Last Observed',
            'Latitude',
            'Longitude',
            'First Observed Type',
            'Will Develop to TC',
            'Developing Date',
        ]
    for col in concate_columns:
        tc_labels[col] = grouped[col].transform(concat_values)

    return tc_labels.drop_duplicates('Date', keep='first')


def load_data(
        data_dir,
        data_shape,
        batch_size=32,
        shuffle=False,
        negative_samples_ratio=None,
        prefetch_batch=1,
        include_tc_position=False,
        subset=None):
    """
    Load data from the given directory.

    :param data_dir: path to directory contains observation data *.nc and label file tc.csv
    :param data_shape: shape of the observation data in this directory.
    :param batch_size: how many observation data should be in a batch.
    :param shuffle: should shuffle data after we have exhausted data points.
    :param negative_samples_ratio: (default: None) the ratio of negative samples to positive samples.
    For instance, if the ratio is 3, then for 1 positive sample, there will be three negative samples.
    If None is passed, all the negative samples are taken.
    :param include_tc_position: whether we should include tc position along with label. Default to False.
    :param subset: allow selecting only a portion of data.
    :returns:
    """
    # Merge observations and labels into single dataframe.
    dataset = load_tc_with_observation_path(data_dir)
    dataset = _filter_negative_samples(dataset, negative_samples_ratio)

    if include_tc_position:
        raise ValueError('Under construction!')
    # FIX: Due to my mistake, the latitude sign is negative, but it should be positive
    # dataset['Latitude'] = -dataset['Latitude'].fillna(0)
    # dataset['Longitude'] = dataset['Longitude'].fillna(0)

    dataset = tf.data.Dataset.from_tensor_slices(
        (dataset['Path'], dataset[['TC', 'Latitude', 'Longitude']] if include_tc_position else dataset['TC']))
    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    # Load given dataset to memory.
    dataset = dataset.map(lambda path, tc: tf.numpy_function(
        lambda x, y: load_observation_data(
            x, y,
            include_tc_position,
            subset=subset),
        inp=[path, tc],
        Tout=([tf.float32, tf.float64]
              if include_tc_position
              else [tf.float32, tf.int64]),
        name='load_observation_data'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Tensorflow should figure out the shape of the output of previous map,
    # but it doesn't, so we have to do it our self.
    # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
    dataset = dataset.map(lambda observation, tc: _set_shape(observation,
                                                             tc,
                                                             data_shape,
                                                             include_tc_position))

    # Cache the dataset for better performance.
    dataset = dataset.cache()

    # Batch the dataset.
    dataset = dataset.batch(batch_size)

    # Always prefetch the data for better performance.
    return dataset.prefetch(prefetch_batch)

def load_data_v1(
        labels_path,
        data_shape,
        batch_size=32,
        shuffle=False,
        negative_samples_ratio=None,
        prefetch_batch=1,
        include_tc_position=False,
        subset=None,
        leadtime: Union[List[int], int] = None,
        group_same_observations=False):
    # Read labels from path.
    labels = pd.read_csv(labels_path)
    
    # Filter in lead time.
    labels = filter_in_leadtime(labels, leadtime)

    # Group same observations.
    # TODO: implement this feature,
    # then rerun all the experiment I did today (Nov 16th, 2021)
    # with val, and test set having group_same_observations=True
    if group_same_observations:
        nb_rows = len(labels)
        labels = group_observations_by_date(labels)
        print(f'Grouping same observations reduces number of rows from {nb_rows} to {len(labels)}.')
        print(labels.head())

    if negative_samples_ratio is not None:
        raise ValueError('Negative samples ratio is not implemented!')

    if include_tc_position:
        raise ValueError('Under Construction!')

    dataset = tf.data.Dataset.from_tensor_slices(
            (labels['Path'], np.where(labels['TC'], 1, 0)))
    
    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    # Load given dataset to memory.
    dataset = dataset.map(lambda path, tc: tf.numpy_function(
        lambda x, y: load_observation_data(
            x, y,
            include_tc_position,
            subset=subset),
        inp=[path, tc],
        Tout=([tf.float32, tf.float64]
              if include_tc_position
              else [tf.float32, tf.int64]),
        name='load_observation_data'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Tensorflow should figure out the shape of the output of previous map,
    # but it doesn't, so we have to do it our self.
    # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
    dataset = dataset.map(lambda observation, tc: _set_shape(observation,
                                                             tc,
                                                             data_shape,
                                                             include_tc_position))

    # Cache the dataset for better performance.
    dataset = dataset.cache()

    # Batch the dataset.
    dataset = dataset.batch(batch_size)

    # Always prefetch the data for better performance.
    return dataset.prefetch(prefetch_batch)


def load_observation_data(observation_path, label, include_tc_position, subset=None):
    dataset = xr.open_dataset(observation_path.decode('utf-8'),
                              engine='netcdf4')

    data = extract_variables_from_dataset(dataset, subset)
    return data, label if include_tc_position else [label]


def load_observation_data_with_tc_probability(observation_path, labels, subset=None):
    dataset = xr.open_dataset(observation_path.decode('utf-8'),
                              engine='netcdf4')
    pass


def _set_shape(observation, label, data_shape, include_tc_position):
    observation.set_shape(data_shape)
    label.set_shape([3] if include_tc_position else [1])
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
    tf.config.set_visible_devices([], 'GPU')
    a = load_data(
        '/N/project/pfec_climo/qmnguyen/tc_prediction/extracted_features/multilevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD/6h_700mb',
        batch_size=1,
        negative_samples_ratio=3,
        data_shape=(41, 181, 13),
        include_tc_position=False)
    a = iter(a)
    for i in range(2):
        # print(i)
        b = next(a)
        print(b)
