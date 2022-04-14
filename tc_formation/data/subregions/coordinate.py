from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class SubregionCoordinate:
    latitudes: np.ndarray
    longitudes: np.ndarray

    size_idx: Tuple[int, int]
    top_left_idx_coord: Tuple[int, int]

    @property
    def top_left_coord(self) -> Tuple[float, float]:
        lat = self.latitudes[self.top_left_idx_coord[0]]
        lon = self.longitudes[self.top_left_idx_coord[1]]
        return (lat, lon)

    @property
    def vertical_slice(self):
        return slice(*self.vertical_range)

    @property
    def horizontal_slice(self):
        return slice(*self.horizontal_range)

    @property
    def area_index(self):
        v, h = self.size_idx
        return v * h

    @property
    def vertical_range(self):
        vert_size = self.size_idx[0]
        vert_start = self.top_left_idx_coord[0]
        return (vert_start, vert_start + vert_size)

    @property
    def horizontal_range(self):
        hor_size = self.size_idx[1]
        hor_start = self.top_left_idx_coord[1]
        return (hor_start, hor_start + hor_size)

    @property
    def vertical_range_deg(self):
        start, end = self.vertical_range
        latitudes = self.latitudes
        return latitudes[start], latitudes[end]

    @property
    def horizontal_range_deg(self):
        start, end = self.horizontal_range
        longitudes = self.longitudes
        return longitudes[start], longitudes[end]
