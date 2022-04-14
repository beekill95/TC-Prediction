from .coordinate import SubregionCoordinate
from .divider import SubRegionDivider
from .utils import IsOceanChecker
from ..tfd_utils import new_py_function
from ..time_series_addons import SingleTimeStepMixin
from .. import utils as data_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from tc_formation.data.time_series import TimeSeriesTropicalCycloneDataLoader
from typing import List, Tuple
import xarray as xr


class SubRegionsTimeSeriesTropicalCycloneDataLoader(TimeSeriesTropicalCycloneDataLoader):
    def __init__(self, subregion_size=(20, 20), subregion_stride=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subregion_size = subregion_size
        self._subregion_stride = subregion_stride

    def _ensure_divider_initialized(self, data_path):
        try:
            return self._divider
        except AttributeError:
            data = xr.load_dataset(data_path, engine='netcdf4')
            self._divider = SubRegionDivider(
                    data['lat'].values,
                    data['lon'].values,
                    self._subregion_size,
                    self._subregion_stride)
            return self._divider

    def _process_to_dataset(self, tc_df: pd.DataFrame, negative_subregions_ratio=None) -> tf.data.Dataset:
        cls = SubRegionsTimeSeriesTropicalCycloneDataLoader

        # Subregion divider.
        divider = self._ensure_divider_initialized(tc_df['Path'].iloc[0][0])
        is_ocean = IsOceanChecker(divider.latitudes, divider.longitudes, ocean_threshold=0.9)
        regions = divider.divide()
        regions = map(lambda r: (
            *r.vertical_range, *r.horizontal_range,
            *r.vertical_range_deg, *r.horizontal_range_deg,
            is_ocean.check(r)), regions)
        regions = list(regions)
        is_region_ocean = [r[-1] for r in regions]
        regions_deg = [tuple(r[4:-1]) for r in regions]
        regions = [tuple(r[:4]) for r in regions]

        dataset = tf.data.Dataset.from_tensor_slices({
            'Path': np.asarray(tc_df['Path'].sum()).reshape((-1, len(self._previous_hours) + 1)),
            'TC': tc_df['TC'],
            'Latitude': tc_df['Latitude'],
            'Longitude': tc_df['Longitude'],
        })

        dataset = dataset.map(
            lambda row: new_py_function(
                    lambda row: cls._load_reanalysis_and_gt(
                            [path.decode('utf-8') for path in row['Path'].numpy()],
                            regions,
                            regions_deg,
                            is_region_ocean,
                            self._subset,
                            row['TC'].numpy(),
                            row['Latitude'].numpy(),
                            row['Longitude'].numpy(),
                            negative_subregions_ratio,
                        ),
                    inp=[row],
                    Tout=[tf.float32, tf.float32],
                    name='load_observation_and_gt',
                ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        dataset = dataset.map(
                lambda X, y: cls._set_dataset_shape(
                    X, y, len(self._previous_hours) + 1, divider.size, self._data_shape[-1]))

        # Unbatch the dataset so it can be batched properly later on.
        dataset = dataset.unbatch()

        return dataset

    def load_single_data(self, data_path):
        return super().load_single_data(data_path)

    @classmethod
    def _load_reanalysis_and_gt(
            cls,
            paths: List[str],
            coords_idx: List[Tuple[int, int, int, int]],
            coords_deg: List[Tuple[float, float, float, float]],
            is_region_ocean: List[bool],
            subset: dict,
            will_form: bool,
            latitude: float,
            longitude: float,
            negative_subregions_ratio: bool) -> Tuple[List[np.ndarray], List[bool]]:
        # assert len(coords_idx) == len(is_region_ocean) 
        # assert len(coords_deg) == len(coords_idx)

        datasets = []
        for path in paths:
            dataset = xr.open_dataset(path, engine='netcdf4')
            dataset = data_utils.extract_variables_from_dataset(dataset, subset)
            datasets.append(np.expand_dims(dataset, axis=0))

        # After this step,
        # the datasets will be a numpy array of shape (time, m, n, features)
        datasets = np.concatenate(datasets, axis=0)
        
        # Divide into sub-regions.
        subregions = cls._divide_into_subregions(datasets, coords_idx)

        # Assign label to each sub-regions.
        subregion_labels = cls._assign_regions_label(coords_deg, will_form, latitude, longitude)

        # Choose subregions.
        subregions, subregion_labels = cls._choose_subregions(subregions, subregion_labels, negative_subregions_ratio)

        # Finally, return the subregions as well as the corresponding labels.
        # The output will have shape of:
        # * subregions: [nb_subregions, time, m, n, features]
        # * subregion_labels: [nb_subregions,]
        return subregions, subregion_labels

    @staticmethod
    def _set_dataset_shape(X, y, nb_time_steps, subregion_idx_size, nb_features):
        X = tf.reshape(X, shape=(-1, nb_time_steps, *subregion_idx_size, nb_features))
        y = tf.reshape(y, shape=(-1, 1))
        return X, y

    @staticmethod
    def _divide_into_subregions(data: np.ndarray, coords_idx: List[Tuple[int, int, int, int]]):
        """
        Divide the data into subregions.
        This function assumes that data has the second and third dimension
        as vertical and horizontal spatial dimensions.
        """
        region_data = []
        for vert_start, vert_end, hor_start, hor_end in coords_idx:
            region_data.append(data[:, vert_start:vert_end, hor_start:hor_end, :])

        return region_data
        
    @classmethod
    def _assign_regions_label(cls, coords_deg: List[Tuple[float, float, float, float]], will_form: bool, latitude: float, longitude: float):
        if not will_form:
            return [False for _ in coords_deg]

        # TODO:
        # should we do a more complex version of this:
        # we could mark the surrounding regions (within some radius from the center of cyclogenesis)
        # as positive?
        return [cls._is_location_in((latitude, longitude), coords)
                for coords in coords_deg]

    @staticmethod
    def _is_location_in(location: Tuple[float, float], coords: Tuple[float, float, float, float]) -> bool:
        lat, lon = location
        lat_start, lat_end, lon_start, lon_end = coords
        return (lat_start <= lat <= lat_end
                and lon_start <= lon <= lon_end)

    @staticmethod
    def _choose_subregions(subregions: List[np.ndarray], labels: List[bool], negative_subregions_ratio: float) -> Tuple[List[np.ndarray], List[bool]]:
        if negative_subregions_ratio is None:
            return subregions, labels

        # Get number of positive and negative subregions.
        label_idx = np.arange(len(labels))
        positives_mask = np.asarray(labels)
        nb_positives = np.sum(positives_mask)
        nb_positives = 1 if nb_positives == 0 else nb_positives
        nb_negatives = nb_positives * negative_subregions_ratio

        # Choose positive and negative subregions.
        negative_idx = label_idx[~positives_mask]
        negative_subregions_idx = np.random.choice(negative_idx, size=nb_negatives)
        positive_subregions_idx = label_idx[positives_mask]
        chosen_idx = np.concatenate([negative_subregions_idx, positive_subregions_idx])
        np.random.shuffle(chosen_idx)

        # Finally, return the chosen subregions.
        return np.asarray(subregions)[chosen_idx], positives_mask[chosen_idx]


class SubRegionsTropicalCycloneDataLoader(SingleTimeStepMixin, SubRegionsTimeSeriesTropicalCycloneDataLoader):
    pass
