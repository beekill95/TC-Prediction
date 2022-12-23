from collections import OrderedDict
import numpy as np
from typing import Union
import xarray as xr


SubsetDict = OrderedDict[str, Union[tuple[float, ...], bool]]


def extract_subset(ds: xr.Dataset, subset: SubsetDict) -> np.ndarray:
    tensors = []
    for key, lev in subset.items():
        values = None
        if isinstance(lev, bool):
            if lev:
                values = ds[key].values
        else:
            values = ds[key].sel(lev=list(lev)).values

        if values is not None:
            if values.ndim == 2:
                values = values[None, ...]

            tensors.append(values)

    tensors = np.concatenate(tensors, axis=0)
    tensors = np.moveaxis(tensors, 0, -1)
    return tensors


def fill_missing_values(ds: xr.Dataset) -> xr.Dataset:
    mean_values = ds.mean(dim=['lat', 'lon'], skipna=True)
    return ds.fillna(mean_values)
