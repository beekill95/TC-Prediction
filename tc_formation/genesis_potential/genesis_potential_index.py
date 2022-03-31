import numpy as np
import xarray as xr


def vorticity_parameter(ds: xr.Dataset) -> np.ndarray:
    vorticity_950 = ds['absvprs'].sel(lev=950).values
    return vorticity_950 * 1e6 + 5.


def corriolis_parameter(ds: xr.Dataset) -> np.ndarray:
    lat = ds.lat
    lon = ds.lon

    # TODO: rotation rate of the Earth.
    omega = 1

    xx, _ = np.meshgrid(lat, lon)
    return 2 * omega * np.sin(xx)


def vertical_shear_parameter(ds: xr.Dataset) -> np.ndarray:
    # Wind field at 950mb.
    u_wind_950 = ds['ugrdprs'].sel(lev=950).values
    v_wind_950 = ds['vgrdprs'].sel(lev=950).values

    # Wind field at 200mb.
    u_wind_200 = ds['ugrdprs'].sel(lev=200).values
    v_wind_200 = ds['vgrdprs'].sel(lev=200).values

    # Vertical wind shear.
    u_vertical = u_wind_200 - u_wind_950
    v_vertical = v_wind_200 - v_wind_950

    # Vertical wind shear magnitude.
    vertical_wind_mag = np.sqrt(u_vertical**2 + v_vertical**2)

    # Return the result.
    return 750.0 / vertical_wind_mag


def ocean_thermal_energy(ds: xr.Dataset) -> np.ndarray:
    # TODO:
    return np.ones((len(ds.lat),  len(ds.lon)))


def moist_stability_parameter(ds: xr.Dataset) -> np.ndarray:
    # TODO:
    return np.ones((len(ds.lat),  len(ds.lon)))


def relative_humidity_parameter(ds: xr.Dataset) -> np.ndarray:
    rh_700_500 = ds['rhprs'].sel(lev=slice(700, 500)).values
    mean_rh = np.mean(rh_700_500, axis=0)
    rh = (mean_rh - 40.0) / 70.0

    return np.clip(rh, 0.0, 1.0)


def thermal_parameter(ds: xr.Dataset) -> np.ndarray:
    ocean_thermal = ocean_thermal_energy(ds)
    moist_stability = moist_stability_parameter(ds)
    rh = relative_humidity_parameter(ds)
    return ocean_thermal * moist_stability * rh


def dynamic_parameter(ds: xr.Dataset) -> np.ndarray:
    vorticity = vorticity_parameter(ds)
    corriolis = corriolis_parameter(ds)
    vertical_shear = vertical_shear_parameter(ds)
    return vorticity * corriolis * vertical_shear


def genesis_potential_index(ds: xr.Dataset) -> np.ndarray:
    return thermal_parameter(ds) * dynamic_parameter(ds)
