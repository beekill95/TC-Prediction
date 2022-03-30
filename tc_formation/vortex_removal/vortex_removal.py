import numpy as np
from typing import Tuple
import xarray as xr


def remove_vortex_ds(dataset: xr.Dataset, centers: np.ndarray, radius: float) -> xr.Dataset:
    dataset = dataset.copy(deep=True)
    minlat = np.min(dataset.lat)
    minlon = np.min(dataset.lon)

    # Translate to pixel coordinates, (0, 0) at top-left corner.
    print(centers)
    centers = np.asarray(centers) - np.asarray([minlat, minlon])
    # print('after', centers)

    for variable, data in dataset.data_vars.items():
        data_values = data.values
        if len(data_values.shape) > 2:
            data_values = np.transpose(data_values, [1, 2, 0])

        processed_data = remove_vortex(data_values, centers, radius)

        if len(processed_data.shape) > 2:
            processed_data = np.transpose(processed_data, [2, 0, 1])

        data = xr.DataArray(processed_data, coords=data.coords, dims=data.dims)
        dataset[variable] = data

    return dataset


def remove_vortex(field: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    """
    Remove tropical cyclones vortex from the field.

    Parameters
    ----------
        field: np.ndarray
            2D observation field.
        centers: np.ndarray
            Position of the tropical cyclone centers.
        radius: float
            Radius of of the TC region to apply the removal algorithm.

    Returns
    -------
    np.ndarray
        The field with the same shape as the original field,
        but with TC removed.
    """
    field = np.copy(field)
    for center in centers:
        x_min, x_max, y_min, y_max = _extract_centered_region_coords(field, center, radius)
        # print(x_min, x_max, y_min, y_max)
        tc_field = field[x_min:x_max, y_min:y_max]
        basic_field = _obtain_basic_field(tc_field)

        # print('inside', np.all(tc_field == basic_field))

        # We further assume that within this region,
        # most of the disturbances are from the TC.
        # Thus, the basic field is enough,
        # no need to further extract the non-hurricane disturbance field.
        field[x_min:x_max, y_min:y_max] = basic_field

        # TODO: apply further smoothing the smooth the edges of the basic field.

    return field


def _extract_centered_region_coords(field: np.ndarray, center: Tuple[float, float], radius: float) -> Tuple[int, int, int, int]:
    """
    Extract coords of a square region surrounding a circle of radius `radius` centered at `center`.

    Returns
    -------
    Tuple[int, int, int, int]
        The coordinate of the left, right, top and bottom border.
    """
    # This will only work with 2D array.
    x_field_max, y_field_max, *_ = np.shape(field)
    x_center, y_center = center

    # Extract the region.
    x_min = max(int(x_center - radius), 0)
    x_max = round(min(x_center + radius, x_field_max))
    
    y_min = max(int(y_center - radius), 0)
    y_max = round(min(y_center + radius, y_field_max))

    return x_min, x_max, y_min, y_max 


def _obtain_basic_field(tc_field: np.ndarray) -> np.ndarray:
    """
    Obtaining basic field as described in the paper by
    [Kurihara et al. 1993](https://journals.ametsoc.org/view/journals/mwre/121/7/1520-0493_1993_121_2030_aisohm_2_0_co_2.xml)
    """
    def apply_filter_first_dim(field: np.ndarray, m: float) -> np.ndarray:
        """This will mutate the field parameter."""
        K = .5 / (1 - np.cos(2 * np.pi / m))
        # original = field
        # print('Original', original)
        # field = np.copy(field)
        field[1:-1] += K * (field[2:] + field[:-2] - 2 * field[1:-1])
        # print('Field', field)
        # print('Inside apply', np.all(original == field))
        return field

    # In the paper,
    # Kurihara shows the procedure as followed:
    #
    # 1. Iteratively smoothing along the zonal direction.
    m_values = [2, 3, 4, 2, 5, 6, 7, 2, 8, 9, 2]
    tc_field = np.copy(_transpose(tc_field))
    for m in m_values:
        # print('zonal', m)
        tc_field = apply_filter_first_dim(tc_field, m)

    # 2. Iteratively smoothing along the meridional direction.
    tc_field = _transpose(tc_field)
    for m in m_values:
        # print('meridional', m)
        tc_field = apply_filter_first_dim(tc_field, m)

    return tc_field

def _transpose(field: np.ndarray):
    return field.T if len(np.shape(field)) == 2 else np.transpose(field, [1, 0, 2])
