#! /usr/bin/env python3

import numpy as np
import xarray as xr

if __name__ == '__main__':
    ds = xr.open_dataset(
        './data_theanh/2020/gdas1.fnl0p25.2020010100.f00.grib2',
        engine='cfgrib',
        backend_kwargs={'indexpath': ''},
        filter_by_keys={'typeOfLevel': 'atmosphere'})
    print(ds, ds.attrs)
