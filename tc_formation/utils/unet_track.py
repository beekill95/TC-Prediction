"""
This module contains essential functionalities to
pinpoint the center of Unet's prediction,
and calculate the probability distribution of those centers.
"""
from __future__ import annotations
from ..metrics import bb

import numpy as np


class UnetPredictionCenter:
    def get_centers(self, prediction, threshold: float = 0.5):
        """
        Get centers of all prediction blob in the given prediction.
        `prediction` is a 2d numpy array, whose values ranges from 0 - 1,
        which represents the probability of TC forming in each cell.
        Returns the detected centers as a list of (lat, lon).
        Note that these `lat` and `lon` are in pixel coordinates,
        from 0, 1, to max(lat) or max(lon).
        Thus, in order to convert back to the real lat, lon,
        you have to index to lat[lat_in_pixel] or lon[lon_in_pixel]
        """
        def calculate_center(box):
            x, y, w, h = box
            # lat, lon = self._origin
            return y + .5 * h, x + .5 * w

        bboxes = bb.extract_bounding_boxes(prediction, threshold)
        centers = list(map(calculate_center, bboxes))
        return centers
        

def tc_formation_spatial_distribution(
        domain_size: tuple[int, int],
        centers: list[tuple[float, float]]):
    tc_counts = np.zeros(domain_size, dtype=np.int32)

    for lat, lon in centers:
        tc_counts[int(lat), int(lon)] += 1

    return tc_counts
