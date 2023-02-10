from __future__ import annotations


import numpy as np
import numpy.typing as npt
from scipy.ndimage.interpolation import geometric_transform


def cartesian_2_polar(img: npt.NDArray[np.float32], order=3):
    """
    This function will convert images in cartesian coordinate into polar coordinate.
    It does this by assuming that the origin of the polar coordinate is the center of the image.
    Also, this will only transform the width and height of the image,
    so if the image is (H, W, C) then the output will be (radius, theta, C)
    """
    h, w, c = img.shape
    w_center = w / 2
    h_center = h / 2

    # Maximum radius is the half of the diagonal.
    max_radius = int(np.sqrt(h*h + w*w) / 2.)

    def _polar_2_cartesian_coords(polar_coords):
        radius, theta, c = polar_coords
        theta = theta * np.pi / 180
        y = h_center - radius * np.sin(theta)
        x = w_center + radius * np.cos(theta)

        return y, x, c

    return geometric_transform(
        img,
        _polar_2_cartesian_coords,
        output_shape=(max_radius, 360, c),
        order=order,
        mode='constant',
        cval=np.nan)


def polar_2_cartesian(img: npt.NDArray[np.float32], original_img_shape: tuple[int, int, int], order=3):
    """
    This will perform the inverse operation of the above.
    """
    assert img.shape[1] == 360, 'Theta must be 360 degrees.'

    h, w, _ = original_img_shape
    h_center, w_center = h / 2, w / 2

    def _cartesian_2_polar_coords(cartesian_coord: tuple[int, int, int]):
        y, x, c = cartesian_coord

        radius = np.sqrt((h_center - y)**2 + (x - w_center)**2)
        theta = np.arctan2(h_center - y, x - w_center)

        # Convert theta to degrees.
        theta = theta * 180. / np.pi
        theta = theta if theta >= 0 else 360 + theta

        # return 1, 1, 1
        return radius, theta, c

    return geometric_transform(
        img,
        _cartesian_2_polar_coords,
        output_shape=original_img_shape,
        order=order,
        mode='nearest')
