from global_land_mask import globe
import numpy as np
from .coordinate import SubregionCoordinate


class IsOceanChecker:
    def __init__(self, latitudes: np.ndarray, longitudes: np.ndarray, ocean_threshold:float = 0.9) -> None:
        self._latitudes = latitudes
        self._longitudes = longitudes
        self._ocean_threshold = ocean_threshold

    @property
    def ocean_mask(self) -> np.ndarray:
        try:
            return self._ocean_mask
        except AttributeError:
            lon = np.where(self._longitudes < 180.0, self._longitudes, 360.0 - self._longitudes)
            yy, xx = np.meshgrid(lon, self._latitudes)
            self._ocean_mask = globe.is_ocean(xx, yy)
            return self._ocean_mask

    def check(self, coord: SubregionCoordinate) -> bool:
        ocean_mask = self.ocean_mask
        subregion = ocean_mask[coord.vertical_slice, coord.horizontal_slice]
        ocean_percentage = np.sum(subregion) / coord.area_index
        return ocean_percentage >= self._ocean_threshold
