import datetime
import glob
import numpy as np
import os
import tensorflow as tf
import xarray as xr

VALIDATION_FROM = datetime.datetime(2016, 1, 1)
TEST_FROM = datetime.datetime(2018, 1, 1)
TIME_DELTA = datetime.timedelta(hours=12)
TIME_FORMAT = 'fnl_%Y%m%d_%H_00'

def _parse_date_from_filename(path):
    filename, _ = os.path.splitext(os.path.basename(path))
    return datetime.datetime.strptime(filename, TIME_FORMAT)

def _convert_date_to_filename(date, dir=None):
    filename = f'{date.strftime(TIME_FORMAT)}.nc'
    return filename if dir is None else os.path.join(dir, filename)

def _is_date_between(date, earlier_date=None, late_date=None):
    assert (earlier_date is not None) or (late_date is not None)
    if earlier_date is None:
        return date < late_date
    elif late_date is None:
        return date >= earlier_date
    else:
        return earlier_date <= date < late_date

def _get_observation_to_reconstruct(observation_path, time_delta):
    date = _parse_date_from_filename(observation_path)
    date += time_delta
    parent_dir = os.path.dirname(observation_path)
    return _convert_date_to_filename(date, parent_dir)

def _list_observation_paths(data_dir):
    return glob.glob(os.path.join(data_dir, 'fnl_*.nc'))

# TODO: should be imported from some data utils.
# Probably never gonna change this,
# but put here so I know how bad I was.
def _extract_variables_from_dataset(dataset: xr.Dataset, subset: dict = None):
    data = []
    for var in dataset.data_vars:
        var = var.lower()
        if subset is not None and var in subset:
            if subset[var] is not None:
                values = dataset[var].sel(lev=subset[var]).values
            else:
                continue
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

def _load_reanalysis(paths, subset: dict):
    def _load_data(path, subset):
        data = xr.open_dataset(path.numpy().decode('utf-8'), engine='netcdf4')
        return _extract_variables_from_dataset(data, subset)
        
    input_path, reconstruct_path = paths
    input_data = _load_data(input_path, subset)
    reconstruct_data = _load_data(reconstruct_path, subset)
    # print('input data shape:', input_data.shape, '\nreconstruct data shape', reconstruct_data.shape)
    return input_data, reconstruct_data

def _set_data_shape(X, Y, data_shape):
    X.set_shape(data_shape)
    Y.set_shape(data_shape)

    return X, Y

def _process_to_dataset(files, time_delta, subset, data_shape):
    files = map(lambda f: (f, _get_observation_to_reconstruct(f, time_delta)), files)
    files = filter(lambda fs: all([os.path.isfile(f) for f in fs]), files)
    files = list(files)
    # print('After filter', len(files))

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(
        lambda f: tf.py_function(
            lambda f: _load_reanalysis(f, subset),
            inp=[f],
            Tout=[tf.float64, tf.float64],
            name='load_reanalysis_input_and_reconstruct',
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    dataset = dataset.map(lambda X, Y: _set_data_shape(X, Y, data_shape))

    return dataset.cache()

def load_reconstruction_datasets(
        data_dir,
        data_shape,
        time_delta=TIME_DELTA,
        validation_from=VALIDATION_FROM,
        test_from=TEST_FROM,
        subset=None):
    files = _list_observation_paths(data_dir)

    # Make sure these files follow ascending order.
    files.sort()

    # Split into training, validation and testing files.
    training = []
    validation = []
    testing = []

    for file in files:
        date = _parse_date_from_filename(file)
        if _is_date_between(date, late_date=validation_from):
            training.append(file)
        elif _is_date_between(date, earlier_date=validation_from, late_date=test_from):
            validation.append(file)
        else:
            testing.append(file)

    # print('Training', len(training), '\nValidation', len(validation), '\nTesting', len(testing))

    # Process these files into dataset and return.
    return (_process_to_dataset(training, time_delta, subset, data_shape),
            _process_to_dataset(validation, time_delta, subset, data_shape),
            _process_to_dataset(testing, time_delta, subset, data_shape))
