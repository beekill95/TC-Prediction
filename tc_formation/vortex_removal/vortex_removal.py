import numpy as np
from typing import Tuple
import xarray as xr


from . import polar_transformations as pt


def remove_vortex_ds(dataset: xr.Dataset, centers: np.ndarray, radius: float) -> xr.Dataset:
    dataset = dataset.copy(deep=True)
    minlat = np.min(dataset.lat)
    minlon = np.min(dataset.lon)

    # Translate to pixel coordinates, (0, 0) at top-left corner.
    centers = np.asarray(centers) - np.asarray([minlat, minlon])

    for variable, data in dataset.data_vars.items():
        # print('Processing variable', variable)
        data_values = data.values
        if len(data_values.shape) > 2:
            data_values = np.transpose(data_values, [1, 2, 0])

        processed_data = remove_vortex(data_values, centers, radius)

        if len(processed_data.shape) > 2:
            processed_data = np.transpose(processed_data, [2, 0, 1])

        data = xr.DataArray(processed_data, coords=data.coords, dims=data.dims)
        dataset[variable] = data
        # break

    return dataset


def remove_vortex(
        field: np.ndarray,
        centers: np.ndarray,
        radius: float,
        min_size: float = 3,
        min_size_for_analyzed_vortex: float = 5) -> np.ndarray:
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
        min_size: float
            Minimum domain size to be considered for vortex removal.
        min_size_for_analyzed_vortex: float
            Minimum domain size to be considered for analyzing vortex.

    Returns
    -------
    np.ndarray
        The field with the same shape as the original field,
        but with TC removed.
    """
    field = np.copy(field)
    for center in centers:
        x_min, x_max, y_min, y_max = _extract_centered_region_coords(field, center, radius)
        if (x_max - x_min < min_size) or (y_max - y_min < min_size):
            continue

        tc_field = field[x_min:x_max, y_min:y_max]
        basic_field = _obtain_basic_field(tc_field)
        # print('== center', center, 'xminmax', x_min, x_max, 'yminmax', y_min, y_max)
        # print('tc_field', tc_field[:5, :5])
        # print('basic field', basic_field[:5, :5])

        # Maybe this assumption was not correct?
        # We further assume that within this region,
        # most of the disturbances are from the TC.
        # Thus, the basic field is enough,
        # no need to further extract the non-hurricane disturbance field.
        # field[x_min:x_max, y_min:y_max] = basic_field

        if (x_max - x_min < min_size_for_analyzed_vortex) or (y_max - y_min < min_size_for_analyzed_vortex):
            field[x_min:x_max, y_min:y_max] = basic_field
            continue

        # TODO: probably in the beginning,
        # the extracted field should be square.
        # Then, if the domain size is too small (probably < 5-7 deg),
        # we can assume that within the domain, most of the disturbance is from the TC,
        # else we can have to subtract the analyzed vortex field.
        disturbance_field = tc_field - basic_field

        # Now, we will obtain the analyzed vortex field.
        # analyzed_vortex_field = _obtain_analyzed_vortex_field(disturbance_field)
        analyzed_vortex_field = _obtain_analyzed_vortex_field_1(disturbance_field)

        # Then, the environmental field is the original field minus the vortex field.
        environmental_field = tc_field - analyzed_vortex_field

        # TODO: apply further smoothing the smooth the edges of the basic field.

        # Finally, reassign the field.
        field[x_min:x_max, y_min:y_max] = environmental_field
        # field[x_min:x_max, y_min:y_max] = disturbance_field
        # field[x_min:x_max, y_min:y_max] = analyzed_vortex_field

    return field


def _extract_centered_region_coords(field: np.ndarray, center: Tuple[float, float], radius: float) -> Tuple[int, int, int, int]:
    """
    Extract coords of a square region surrounding a circle of radius `radius` centered at `center`.

    Returns
    -------
    Tuple[int, int, int, int]
        The coordinate of the left, right, top and bottom border.
    """
    def adjust_end_points_to_center(lower, upper, center):
        min_dist = min(upper - center, center - lower)
        return int(center - min_dist), int(round(center + min_dist))

    # This will only work with 2D array.
    x_field_max, y_field_max, *_ = np.shape(field)
    x_center, y_center = center

    # Extract the region.
    x_min = max(int(x_center - radius), 0)
    x_max = round(min(x_center + radius, x_field_max))
    x_min, x_max = adjust_end_points_to_center(x_min, x_max, x_center)

    y_min = max(int(y_center - radius), 0)
    y_max = round(min(y_center + radius, y_field_max))
    y_min, y_max = adjust_end_points_to_center(y_min, y_max, y_center)

    return x_min, x_max, y_min, y_max 


def _obtain_basic_field(tc_field: np.ndarray) -> np.ndarray:
    """
    Obtaining basic field as described in the paper by
    [Kurihara et al. 1993](https://journals.ametsoc.org/view/journals/mwre/121/7/1520-0493_1993_121_2030_aisohm_2_0_co_2.xml)
    """
    def apply_filter_first_dim(field: np.ndarray, m: float) -> np.ndarray:
        """This will mutate the field parameter."""
        K = .5 / (1 - np.cos(2 * np.pi / m))
        field[1:-1] += K * (field[2:] + field[:-2] - 2 * field[1:-1])
        return field

    # In the paper,
    # Kurihara shows the procedure as followed:
    #
    # 1. Iteratively smoothing along the zonal direction.
    m_values = [2, 3, 4, 2, 5, 6, 7, 2, 8, 9, 2]
    tc_field = np.copy(_transpose(tc_field))
    for m in m_values:
        tc_field = apply_filter_first_dim(tc_field, m)

    # 2. Iteratively smoothing along the meridional direction.
    tc_field = _transpose(tc_field)
    for m in m_values:
        tc_field = apply_filter_first_dim(tc_field, m)

    # why I didn't transpose again?
    # Now I do, but why I didn't do it before?
    # No, you shouldn't, because you've transposed before doing all these stuffs.
    return tc_field


def _obtain_analyzed_vortex_field_1(disturbance_field: np.ndarray) -> np.ndarray:
    def gauss_kernel(field, var):
        h, w, _ = field.shape
        yy, xx = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        yy -= h / 2.
        xx -= w / 2.
        kernel = np.exp(-(xx**2 + yy**2)/(2*var))# / (2 * np.pi * var)
        return kernel[..., None]

    if disturbance_field.ndim == 2:
        has_2_dim = True
        disturbance_field = disturbance_field[..., None]
    else:
        has_2_dim = False

    kernel = gauss_kernel(disturbance_field, var=16)
    # print('kernel', kernel[..., 0])
    analyzed_vortex = disturbance_field * kernel
    return analyzed_vortex if not has_2_dim else analyzed_vortex[..., 0]


def _obtain_analyzed_vortex_field(disturbance_field: np.ndarray) -> np.ndarray:
    if disturbance_field.ndim == 2:
        has_2_dim = True
        disturbance_field = disturbance_field[..., None]
    else:
        has_2_dim = False

    w, h, _ = disturbance_field.shape
    disturbance_field_polar = pt.cartesian_2_polar(disturbance_field)
    r = np.arange(disturbance_field_polar.shape[0])

    # r0 will be defined as half of the domain radius.
    # TODO: implement automatic r0 detection.
    # r0 = min(w // 2, h // 2) // 2
    r0 = min(w // 2, h // 2)
    l = r0 / 5.

    # Adjust r.
    r = np.where(r > r0, r0, r)

    print('w', w, 'h', h)
    print('cartesian', disturbance_field[:5, :5])
    print('polar', disturbance_field_polar[:5, :5])
    print(disturbance_field_polar[r0, :5])

    # hd_bar has shape (theta, c)
    hD_bar = np.nansum(disturbance_field_polar[r0], axis=0) / (2 * np.pi)
    # E_r has shape (r,)
    E_r = ((np.exp(-(r0 - r)**2 / l**2) - np.exp(-r0**2/l**2))
            / (1. - np.exp(-r0**2/l**2)))
    print(f'{r0=},\n{l=},\n{hD_bar=},\n{E_r=},\n')

    # Calculate the analyzed vortex field.
    analyzed_vortex_polar = disturbance_field_polar - (
        disturbance_field_polar[r0:r0+1]*E_r[..., None, None]
        + hD_bar[None, ...]*(1 - E_r[..., None, None]))

    # Return the analyzed vortex in the original coordinate.
    print('analyzed vortex polar', analyzed_vortex_polar[:5, :5])
    analyzed_vortex = pt.polar_2_cartesian(
        analyzed_vortex_polar,
        disturbance_field.shape,
        order=1)
    print('analyzed vortex', analyzed_vortex[:5, :5, 0])#, analyzed_vortex_polar[1, 1, 1])

    return analyzed_vortex if not has_2_dim else analyzed_vortex[..., 0]


def _transpose(field: np.ndarray):
    return field.T if len(np.shape(field)) == 2 else np.transpose(field, [1, 0, 2])
