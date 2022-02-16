from datetime import datetime, timedelta
from functools import reduce, partial
import glob
import tc_formation.data.label as label
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tc_formation.data import utils
import tensorflow as tf
from typing import Union, List
import xarray as xr


def _extract_date_from_observation_path(path):
    filename = Path(path).stem
    return ''.join(filename.split('_')[1:-1])

"""DEPRECATED: should favor the same method in label module."""
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
            if subset[var] is not None:
                values = dataset[var].sel(lev=subset[var]).values
            else:
                continue
            # print(var, subset[var])
        else:
            values = dataset[var].values
            # print(var)

        # For 2D dataarray, make it 3D.
        if len(np.shape(values)) != 3:
            values = np.expand_dims(values, 0)

        data.append(values)

    # Reshape data so that it have channel_last format.
    data = np.concatenate(data, axis=0)
    data = np.moveaxis(data, 0, -1)

    return data

"""DEPRECATED: should favor the same method in label module."""
def filter_in_leadtime(tc: pd.DataFrame, leadtimes: Union[List[int], int] = None):
    if leadtimes is None:
        return tc

    if not isinstance(leadtimes, list):
        leadtimes = [leadtimes]

    # First, we will keep all negative cases.
    mask = ~tc['TC']

    # Then, loop through each lead time to get observations that belong to that leadtime.
    observation_dates = parse_tc_datetime(tc['Date'])
    tc_first_observed_dates = parse_tc_datetime(tc['First Observed'])
    for leadtime in leadtimes:
        leadtime = timedelta(hours=leadtime)
        mask |= (tc_first_observed_dates - observation_dates) == leadtime

    return tc[mask]

"""DEPRECATED: should favor the same method in label module."""
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
        other_happening_tc_ratio=None,
        prefetch_batch=1,
        include_tc_position=False,
        subset=None,
        leadtime: Union[List[int], int] = None,
        group_same_observations=False):
    # Read labels from path.
    labels = pd.read_csv(labels_path)

    # Make sure that when other happening tc ratio is set,
    # the our labels have the information to do so.
    if other_happening_tc_ratio is not None:
        assert ('Is Other TC Happening' in labels.columns), 'Incompatible label version, requires labels version at least v3'
    
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

    print(f'Number of positive labels: {np.sum(labels["TC"])}')
    print(f'Number of negative labels: {np.sum(~labels["TC"])}')

    labels = _filter_negative_samples(labels, negative_samples_ratio, other_happening_tc_ratio)

    if include_tc_position:
        raise ValueError('Under Construction!')

    dataset = tf.data.Dataset.from_tensor_slices((labels['Path'], np.where(labels['TC'], 1, 0)))
    
    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    # Load given dataset to memory.
    dataset = dataset.map(lambda path, tc: tf.numpy_function(
        partial(load_observation_data_v1, subset=subset),
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
    dataset = dataset.map(
            lambda observation, tc: _set_shape(observation,
                                               tc,
                                               data_shape,
                                               include_tc_position))

    # Cache the dataset for better performance.
    dataset = dataset.cache()

    # Batch the dataset.
    dataset = dataset.batch(batch_size)

    # Always prefetch the data for better performance.
    return dataset.prefetch(prefetch_batch)

def load_data_with_tc_probability(
        labels_path,
        data_shape,
        batch_size=32,
        shuffle=False,
        prefetch_batch=1,
        subset=None,
        tc_avg_radius_lat_deg=2,
        leadtime: Union[List[int], int] = None):
    # Read labels from path.
    labels = pd.read_csv(labels_path, dtype={
        'TC Id': str,
        'First Observed': str,
        'Last Observed': str,
        'First Observed Type': str,
        'Will Develop to TC': str,
        'Developing Date': str,
    })

    # Filter in lead time.
    labels = filter_in_leadtime(labels, leadtime)
    labels = group_observations_by_date(labels)

    # dataset = tf.data.Dataset.from_tensor_slices( (labels['Path'], np.where(labels['TC'], 1, 0)))
    dataset = tf.data.Dataset.from_tensor_slices(dict(labels[['Path', 'TC', 'Latitude', 'Longitude']]))
    
    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    # Load given dataset to memory.
    dataset = dataset.map(lambda row: utils.new_py_function(
        partial(load_observation_data_with_tc_probability, subset=subset, tc_avg_radius_lat_deg=tc_avg_radius_lat_deg),
        inp=[row],
        Tout=[tf.float32, tf.float32],
        name='load_observation_data'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Tensorflow should figure out the shape of the output of previous map,
    # but it doesn't, so we have to do it our self.
    # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
    dataset = dataset.map(partial(_set_shape_tc_probability, data_shape=data_shape))

    # Cache the dataset for better performance.
    dataset = dataset.cache()

    # Batch the dataset.
    dataset = dataset.batch(batch_size)

    # Always prefetch the data for better performance.
    return dataset.prefetch(prefetch_batch)


def load_time_series_dataset(
        label_path,
        data_shape,
        batch_size=32,
        prefetch_batch=1,
        leadtimes=None,
        shuffle=False,
        group_observations_by_date=True,
        tc_avg_radius_lat_deg=2,
        subset=None):
    # TODO: change the name of this!
    def a(path: str) -> [str]:
        name = os.path.basename(path)
        date = datetime.strptime(''.join(list(name)[4:-3]), '%Y%m%d_%H_%M')
        date -= timedelta(hours=6)
        new_path = os.path.join(
                os.path.dirname(path),
                f"fnl_{date.strftime('%Y%m%d_%H_%M')}.nc")
        return [path, new_path]

    def files_exist(row) -> bool:
        paths = [p.decode('utf-8') for p in row['Path'].numpy()]
        return all([os.path.isfile(p) for p in paths])

    def b_with_tc_prob(row, tc_avg_radius_lat_deg=2, clip_threshold=0.1, subset=None):
        paths = [p.decode('utf-8') for p in row['Path'].numpy()]

        ds1 = xr.open_dataset(paths[0], engine='netcdf4')
        ds2 = xr.open_dataset(paths[1], engine='netcdf4')

        data1 = extract_variables_from_dataset(ds1, subset)
        data2 = extract_variables_from_dataset(ds2, subset)
        diff = data1 - data2

        data = np.concatenate([data1, diff], axis=-1)

        # Just for testing!
        # TODO: refactor this into a separate function,
        # so that we can reuse this function in multiple places.
        groundtruth = np.zeros(data.shape[:-1])

        latitudes = ds1['lat']
        longitudes = ds1['lon']
        x, y = np.meshgrid(longitudes, latitudes)
        if row['TC']:
            lats = row['Latitude'].numpy()
            lons = row['Longitude'].numpy()
            lats = lats if isinstance(lats, list) else [lats]
            lons = lons if isinstance(lons, list) else [lons]
            for lat, lon in zip(lats, lons):
                x_diff = x - lon
                y_diff = y - lat

                # RBF kernel.
                prob = np.exp(-(x_diff * x_diff + y_diff * y_diff)/(2 * tc_avg_radius_lat_deg ** 2))
                # prob = np.exp(-(x_diff * x_diff + y_diff * y_diff) / tc_avg_radius_lat_deg)
                prob[prob < clip_threshold] = 0
                groundtruth += prob

        groundtruth = groundtruth[:, :, None]
        groundtruth = np.where(groundtruth > 0, 1, 0)

        return data, groundtruth

    tc_labels = label.load_label(label_path, group_observations_by_date, leadtimes)
    tc_labels['Path'] = tc_labels['Path'].apply(a)
    # dataset = tf.data.Dataset.from_tensor_slices(dict(tc_labels[['Path', 'TC', 'Latitude', 'Longitude']]))
    dataset = tf.data.Dataset.from_tensor_slices({
        'Path': np.asarray(tc_labels['Path'].sum()).reshape((-1, 2)),
        'TC': tc_labels['TC'],
        'Latitude': tc_labels['Latitude'],
        'Longitude': tc_labels['Longitude'],
    })

    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    # Load given dataset to memory.
    dataset = dataset.filter(lambda row: utils.new_py_function(
        files_exist,
        inp=[row],
        Tout=bool,
        name='files_exist_filter'))
    dataset = dataset.map(lambda row: utils.new_py_function(
        partial(b_with_tc_prob, subset=subset, tc_avg_radius_lat_deg=tc_avg_radius_lat_deg),
        inp=[row],
        Tout=[tf.float32, tf.float32],
        name='load_observation_data'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Tensorflow should figure out the shape of the output of previous map,
    # but it doesn't, so we have to do it our self.
    # https://github.com/tensorflow/tensorflow/issues/31373#issuecomment-524666365
    dataset = dataset.map(partial(_set_shape_tc_probability, data_shape=data_shape))

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

def load_observation_data_v1(path, tc, subset=None):
    # print(path)
    dataset = xr.open_dataset(path.decode('utf-8'), engine='netcdf4')
    data = extract_variables_from_dataset(dataset, subset)
    return data, [tc]

def load_observation_data_with_tc_probability(
        row,
        tc_avg_radius_lat_deg=2,
        clip_threshold=0.1,
        subset=None,
        sigmoid_output=True):
    path = row['Path'].numpy().decode('utf-8')
    dataset = xr.open_dataset(path, engine='netcdf4')
    data = extract_variables_from_dataset(dataset, subset)
    
    groundtruth = np.zeros(data.shape[:-1])

    latitudes = dataset['lat']
    longitudes = dataset['lon']
    x, y = np.meshgrid(longitudes, latitudes)
    if row['TC']:
        lats = row['Latitude'].numpy()
        lons = row['Longitude'].numpy()
        lats = lats if isinstance(lats, list) else [lats]
        lons = lons if isinstance(lons, list) else [lons]
        for lat, lon in zip(lats, lons):
            x_diff = x - lon
            y_diff = y - lat

            # RBF kernel.
            prob = np.exp(-(x_diff * x_diff + y_diff * y_diff)/(2 * tc_avg_radius_lat_deg ** 2))
            # prob = np.exp(-(x_diff * x_diff + y_diff * y_diff) / tc_avg_radius_lat_deg)
            prob[prob < clip_threshold] = 0
            groundtruth += prob

    if sigmoid_output:
        new_groundtruth = np.zeros(np.shape(groundtruth) + (1,))
        new_groundtruth[:, :, 0] = np.where(groundtruth > 0, 1, 0)
    else:
        new_groundtruth = np.zeros(np.shape(groundtruth) + (2,))
        new_groundtruth[:, :, 0] = np.where(groundtruth == 0, 1, 0)
        new_groundtruth[:, :, 1] = np.where(groundtruth > 0, 1, 0)

    return data, new_groundtruth


def _set_shape(observation, label, data_shape, include_tc_position):
    observation.set_shape(data_shape)
    label.set_shape([3] if include_tc_position else [1])
    return observation, label

def _set_shape_tc_probability(observation, prob, data_shape):
    observation.set_shape(data_shape)
    prob.set_shape(data_shape[:2] + (1,))
    return observation, prob


def _filter_negative_samples(dataset, negative_samples_ratio=None, other_happening_tc_ratio=None):
    if negative_samples_ratio is None and other_happening_tc_ratio is None:
        return dataset

    positive_samples = dataset[dataset['TC'] == 1]
    samples = [positive_samples]
    print(f'Positive samples: {len(positive_samples)}')

    negative_samples = dataset[(dataset['TC'] == 0) & (dataset['Is Other TC Happening'] == 0)]
    if negative_samples_ratio is not None:
        nb_negative_samples_to_take = int(
            len(positive_samples) * negative_samples_ratio)
        negative_samples = negative_samples.sample(nb_negative_samples_to_take)
        samples.append(negative_samples)
    else:
        samples.append(negative_samples)

    print(f'Negative samples: {len(negative_samples)}')

    other_happening_tc_samples = dataset[(dataset['TC'] == 0) & (dataset['Is Other TC Happening'] == 1)]
    if other_happening_tc_ratio is not None:
        nb_other_happening_tc_to_take = int(len(positive_samples) * other_happening_tc_ratio)
        other_happening_tc_samples = other_happening_tc_samples.sample(nb_other_happening_tc_to_take)
        samples.append(other_happening_tc_samples)
    else:
        samples.append(other_happening_tc_samples)

    print(f'Other happening TC samples: {len(other_happening_tc_samples)}')

    result = pd.concat(samples)
    print(f'Total samples: {len(result)}')
    return result.sort_values(by='First Observed').reset_index()


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
