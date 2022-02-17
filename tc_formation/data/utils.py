import numpy as np
import xarray as xr

def extract_variables_from_dataset(dataset: xr.Dataset, subset: dict = None):
    data = []
    for var in dataset.data_vars:
        var = var.lower()
        if subset is not None and var in subset:
            if subset[var] is not None:
                values = dataset[var].sel(lev=subset[var]).values
            else:
                continue
        else:
            values = dataset[var].values

        # For 2D dataarray, make it 3D.
        if len(np.shape(values)) != 3:
            values = np.expand_dims(values, 0)

        data.append(values)

    # Reshape data so that it have channel_last format.
    data = np.concatenate(data, axis=0)
    data = np.moveaxis(data, 0, -1)

    return data

