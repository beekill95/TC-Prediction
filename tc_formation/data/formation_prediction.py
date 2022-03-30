from ast import literal_eval
import numpy as np
import pandas as pd
from tc_formation.data.time_series import TimeSeriesTropicalCycloneDataLoader
from tc_formation.data.time_series_addons import SingleTimeStepMixin
import tc_formation.data.utils as data_utils
import tc_formation.data.tfd_utils as tfd_utils
import tensorflow as tf
from typing import List, Tuple
import xarray as xr

class TimeSeriesTCFormationDataLoader(TimeSeriesTropicalCycloneDataLoader):
    def __init__(self, data_shape, previous_hours: List[int], subset=None, produce_other_tc_locations_mask=False, tc_avg_radius_lat_deg=3, clip_threshold=0.1):
        super().__init__(data_shape, previous_hours=previous_hours, subset=subset)
        
        self._produce_other_tc_locations_mask = produce_other_tc_locations_mask
        self._tc_avg_radius_lat_deg = tc_avg_radius_lat_deg
        self._clip_threshold = clip_threshold

    def _process_to_dataset(self, tc_df: pd.DataFrame) -> tf.data.Dataset:
        # This will only works with labels v4+
        if self._produce_other_tc_locations_mask:
            assert 'Other TC Locations' in tc_df.columns, 'Producing other TCs locations requires labels v4+'

            # Convert from string to list.
            tc_df['Other TC Locations'] = tc_df['Other TC Locations'].apply(literal_eval).apply(lambda x: np.asarray(x, dtype=np.float64))

        cls = TimeSeriesTCFormationDataLoader

        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': np.asarray(tc_df['Path'].sum()).reshape((-1, len(self._previous_hours) + 1)),
            'TC': tc_df['TC'],
            'Latitude': tc_df['Latitude'],
            'Longitude': tc_df['Longitude'],
            'Other TC Locations': tf.ragged.constant(tc_df['Other TC Locations']) if self._produce_other_tc_locations_mask else None,
        })
        print('Dataset created ...')

        dataset = dataset.map(
            lambda row: tfd_utils.new_py_function(
                    lambda row: cls._load_reanalysis_gt_and_mask(
                            [path.decode('utf-8') for path in row['Path'].numpy()],
                            self._subset,
                            row['TC'].numpy(),
                            self._data_shape,
                            self._produce_other_tc_locations_mask,
                            row['Other TC Locations'].numpy(),
                            self._tc_avg_radius_lat_deg,
                            self._clip_threshold,
                        ),
                    inp=[row],
                    Tout=[tf.float32, tf.float32, tf.float32],
                    name='load_reanalysis_gt_and_other_tc_mask',
                ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Set the output shape.
        dataset = dataset.map(
                lambda X, Y, mask: cls._set_shape(X, Y, mask, len(self._previous_hours) + 1, self._data_shape))

        # If we don't need the locations mask,
        # just remove it from the output.
        if not self._produce_other_tc_locations_mask:
            print('Remove other tc locations mask from output ...')
            dataset = dataset.map(lambda X, Y, _: (X, Y))

        print('DONE creating dataset.')
        return dataset

    def load_single_data(self, data_path: List[str], has_tc: bool, other_tc_locations: List[Tuple[float, float]]):
        cls = TimeSeriesTCFormationDataLoader
        data = cls._load_reanalysis_gt_and_mask(
            data_path,
            self._subset,
            has_tc=has_tc,
            data_shape=self._data_shape,
            produce_mask=self._produce_other_tc_locations_mask,
            other_tc_locations=other_tc_locations,
            tc_avg_radius_lat_deg=self._tc_avg_radius_lat_deg,
            clip_threshold=self._clip_threshold,
        )

        if not self._produce_other_tc_locations_mask:
            print('Remove other tc locations mask from output ...')
            data = data[:-1]

        return data

    @classmethod
    def _load_reanalysis_gt_and_mask(
        cls, 
        paths: List[str],
        subset: dict,
        has_tc: bool,
        data_shape: Tuple[int, int, int],
        produce_mask: bool,
        other_tc_locations: List[Tuple[float, float]],
        tc_avg_radius_lat_deg: int = 3,
        clip_threshold: float = 0.1,
    ):
        assert len(paths) > 0, 'Paths should have at least one element!'

        datasets = []
        for path in paths:
            dataset = xr.open_dataset(path, engine='netcdf4')
            latitudes = dataset['lat']
            longitudes = dataset['lon']
            dataset = data_utils.extract_variables_from_dataset(dataset, subset)
            datasets.append(np.expand_dims(dataset, axis=0))
        datasets = np.concatenate(datasets, axis=0)

        mask = cls._create_other_tc_locations_mask(
            produce_mask,
            data_shape,
            latitudes,
            longitudes,
            other_tc_locations,
            tc_avg_radius_lat_deg,
            clip_threshold,
        )

        return datasets, np.asarray([has_tc], dtype=np.float64), mask
        
    @classmethod
    def _create_other_tc_locations_mask(
        cls,
        produce_mask: bool,
        data_shape: Tuple[int, int, int],
        data_latitudes: np.ndarray,
        data_longitudes: np.ndarray,
        other_tc_locations: List[Tuple[float, float]],
        tc_avg_radius_lat_deg: int = 3,
        clip_threshold: float = 0.1,
     ):
        # If we don't want to produce mask,
        # then just create a dummy mask where all grid points are accepted.
        if not produce_mask:
            return np.ones_like(data_shape[:-1], dtype=np.float64)

        has_other_tc_mask = np.zeros(data_shape[:-1], dtype=np.float64)
        xx, yy = np.meshgrid(data_longitudes, data_latitudes)
        
        for lat, lon in other_tc_locations:
            x_diff = xx - lon
            y_diff = yy - lat

            # RBF kernel.
            prob = np.exp(-(x_diff * x_diff + y_diff * y_diff) / (2 * tc_avg_radius_lat_deg ** 2))
            prob[prob < clip_threshold] = 0

            has_other_tc_mask += prob

        mask = np.where(has_other_tc_mask >= clip_threshold, 0.0, 1.0)
        return np.expand_dims(mask, axis=-1)

    @classmethod
    def _set_shape(cls, X, Y, mask, nb_previous_hours, data_shape):
        X.set_shape((nb_previous_hours,) + data_shape)
        Y.set_shape([1])
        mask.set_shape(data_shape[:-1] + (1,))

        return X, Y, mask


class TCFormationPredictionDataLoader(SingleTimeStepMixin, TimeSeriesTCFormationDataLoader):
    pass

class TimeSeriesFocusedTCFormationDataLoader(TimeSeriesTropicalCycloneDataLoader):
    def __init__(self, data_shape, previous_hours: List[int]=[], subset=None, tc_avg_radius_lat_deg=3, clip_threshold=0.1, easy=False):
        super().__init__(data_shape, previous_hours=previous_hours, subset=subset)

        self._tc_avg_radius_lat_deg = tc_avg_radius_lat_deg
        self._clip_threshold = clip_threshold

        # Not currently using in any where.
        # But the intention is if this flag is False,
        # then we don't care where the focus area is,
        # i.e. the area doesn't have to avoid the other currently happening TCs.
        self._easy = easy

    def _process_to_dataset(self, tc_df: pd.DataFrame) -> tf.data.Dataset:
        # This will only works with labels v4+
        if self._easy:
            assert 'Other TC Locations' in tc_df.columns, 'Easy construction requires labels v4+'

            # Convert from string to list.
            tc_df['Other TC Locations'] = tc_df['Other TC Locations'].apply(literal_eval).apply(lambda x: np.asarray(x, dtype=np.float64))

        cls = TimeSeriesFocusedTCFormationDataLoader

        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': np.asarray(tc_df['Path'].sum()).reshape((-1, len(self._previous_hours) + 1)),
            'TC': tc_df['TC'],
            'Latitude': tc_df['Latitude'],
            'Longitude': tc_df['Longitude'],
            'Other TC Locations': tf.ragged.constant(tc_df['Other TC Locations']) # if self._easy else None
        })
        print('Dataset created ...')

        dataset = dataset.map(
            lambda row: tfd_utils.new_py_function(
                    lambda row: cls._load_reanalysis_gt_and_focused_mask(
                            [path.decode('utf-8') for path in row['Path'].numpy()],
                            self._subset,
                            row['TC'].numpy(),
                            self._data_shape,
                            [[row['Latitude'].numpy(), row['Longitude'].numpy()]],
                            self._tc_avg_radius_lat_deg,
                            self._clip_threshold,
                        ),
                    inp=[row],
                    Tout=[tf.float32, tf.float32, tf.float32],
                    name='load_reanalysis_gt_and_other_tc_mask',
                ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        # Set the output shape.
        dataset = dataset.map(
                lambda X, Y, mask: cls._set_shape(X, Y, mask, len(self._previous_hours) + 1, self._data_shape))

        print('DONE creating dataset.')
        return dataset

    def load_single_data(self, data_paths: List[str], has_tc: bool, latitude: float, longitude: float):
        cls = TimeSeriesFocusedTCFormationDataLoader

        data = cls._load_reanalysis_gt_and_focused_mask(
            data_paths,
            subset=self._subset,
            has_tc=has_tc,
            data_shape=self._data_shape,
            tc_locations=[(latitude, longitude)],
            tc_avg_radius_lat_deg=self._tc_avg_radius_lat_deg,
            clip_threshold=self._clip_threshold,
        )

        return data

    @classmethod
    def _load_reanalysis_gt_and_focused_mask(
        cls,
        paths: List[str],
        subset: dict,
        has_tc: bool,
        data_shape: Tuple[int, int, int],
        tc_locations: List[Tuple[float, float]],
        tc_avg_radius_lat_deg: int = 3,
        clip_threshold: float = 0.1,
    ):
        assert len(paths) > 0, 'Paths should have at least one element!'

        datasets = []
        for path in paths:
            dataset = xr.open_dataset(path, engine='netcdf4')
            latitudes = dataset['lat']
            longitudes = dataset['lon']
            dataset = data_utils.extract_variables_from_dataset(dataset, subset)
            datasets.append(np.expand_dims(dataset, axis=0))
        datasets = np.concatenate(datasets, axis=0)

        mask = cls._create_tc_locations_mask(
            data_shape,
            latitudes,
            longitudes,
            tc_locations,
            tc_avg_radius_lat_deg,
            clip_threshold,
        ) if has_tc else cls._create_fake_focused_mask_for_non_TC_observation(
            latitudes,
            longitudes,
            tc_avg_radius_lat_deg,
            clip_threshold,
        )

        return datasets, np.asarray([has_tc], dtype=np.float64), mask

    @classmethod
    def _create_tc_locations_mask(
        cls,
        data_shape: Tuple[int, int, int],
        data_latitudes: np.ndarray,
        data_longitudes: np.ndarray,
        tc_locations: List[Tuple[float, float]],
        tc_avg_radius_lat_deg: int = 3,
        clip_threshold: float = 0.1,
     ):
        tc_mask = np.zeros(data_shape[:-1], dtype=np.float64)
        xx, yy = np.meshgrid(data_longitudes, data_latitudes)
        
        for lat, lon in tc_locations:
            x_diff = xx - lon
            y_diff = yy - lat

            # RBF kernel.
            prob = np.exp(-(x_diff * x_diff + y_diff * y_diff) / (2 * tc_avg_radius_lat_deg ** 2))
            prob[prob < clip_threshold] = 0

            tc_mask += prob

        mask = np.where(tc_mask >= clip_threshold, 1.0, 0.0)
        return np.expand_dims(mask, axis=-1)

    @classmethod
    def _create_fake_focused_mask_for_non_TC_observation(
        cls,
        data_latitudes: np.ndarray,
        data_longitudes: np.ndarray,
        tc_avg_radius_lat_deg: int = 3,
        clip_threshold: float = 0.1,
    ):
        xx, yy = np.meshgrid(data_longitudes, data_latitudes)

        # Randomly create focused region.
        min_lat, max_lat = np.min(data_latitudes) + 20, np.max(data_latitudes) - 5
        min_lon, max_lon = np.min(data_longitudes) + 5, np.max(data_longitudes) - 5
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)

        # Create mask.
        x_diff = xx - lon
        y_diff = yy - lat
        prob = np.exp(-(x_diff * x_diff + y_diff * y_diff) / (2 * tc_avg_radius_lat_deg ** 2))

        mask = np.where(prob >= clip_threshold, 1.0, 0.0)
        return np.expand_dims(mask, axis=-1)

    @classmethod
    def _set_shape(cls, X, Y, mask, nb_previous_hours, data_shape):
        X.set_shape((nb_previous_hours,) + data_shape)
        Y.set_shape([1])
        mask.set_shape(data_shape[:-1] + (1,))

        return X, Y, mask


class FocusedTCFormationDataLoader(SingleTimeStepMixin, TimeSeriesFocusedTCFormationDataLoader):
    pass
