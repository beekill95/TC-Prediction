import numpy as np
from typing import Tuple
from .coordinate import SubregionCoordinate


class SubRegionDivider:
    def __init__(self,
            latitudes: np.ndarray,
            longitudes: np.ndarray,
            subregion_size: Tuple[int, int],
            subregion_stride: int) -> None:
        self._latitudes = latitudes
        self._longitudes = longitudes
        self._size_deg = subregion_size
        self._stride_deg = subregion_stride

    @property
    def longitudes(self):
        return self._longitudes

    @property
    def latitudes(self):
        return self._latitudes

    @property
    def stride(self):
        try:
            return self._stride
        except AttributeError:
            degrees = (self._longitudes
                       if len(self._longitudes) > len(self._latitudes)
                       else self._latitudes)
            self._stride = _try_convert_degree_diff_to_index_diff(degrees, self._stride_deg)
            return self._stride

    @property
    def size(self):
        try:
            return self._size
        except AttributeError:
            vert_size = _try_convert_degree_diff_to_index_diff(self._latitudes, self._size_deg[0])
            hor_size = _try_convert_degree_diff_to_index_diff(self._longitudes, self._size_deg[1])
            self._size = (vert_size, hor_size)
            return self._size

    def divide(self):
        """
        Divide a large region into subregions
        whose size is specified in the constructor.
        The return value is a list of coordinates (in the index domain) of the top-left corner.
        So, in order to recover the original latitude and longitude,
        you should use latitudes[ith_vert] or longitudes[jth_hor].
        """
        vert_max = len(self._latitudes) - self.size[0]
        hor_max = len(self._longitudes) - self.size[1]
        for ith_vert in range(0, vert_max, self.stride):
            for jth_hor in range(0, hor_max, self.stride):
                top_left = (ith_vert, jth_hor)
                yield self._create_subregion_coord(top_left)

    def _create_subregion_coord(self, top_left_idx):
        return SubregionCoordinate(latitudes=self._latitudes,
                                   longitudes=self._longitudes,
                                   size_idx=self._size,
                                   top_left_idx_coord=top_left_idx)


def _try_convert_degree_diff_to_index_diff(degrees, deg_diff):
    index = None

    for i in range(1, len(degrees)):
        diff = degrees[i:] - degrees[:-i]
        if np.allclose(deg_diff, diff):
            index = i
            break

    assert index is not None, 'Cannot find suitable index difference!'
    return index
