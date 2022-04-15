from global_land_mask import globe
import numpy as np
import xarray as xr


def vorticity_parameter(ds: xr.Dataset) -> np.ndarray:
    vorticity_950 = ds['absvprs'].sel(lev=950).values
    # TODO: do we need to multiply with 1e6?
    return vorticity_950 + 5.


def corriolis_parameter(ds: xr.Dataset) -> np.ndarray:
    lat = ds.lat
    lon = ds.lon

    # TODO: rotation rate of the Earth.
    omega = 1

    _, yy = np.meshgrid(lon, lat)
    return 2 * omega * np.sin(yy * np.pi / 180.0)


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
    vertical_wind_mag = np.sqrt(u_vertical**2 )# + v_vertical**2)

    return vertical_wind_mag

    # Return the result.
    # return 1. / (vertical_wind_mag / 750.0 + 3.)


def ocean_thermal_energy(ds: xr.Dataset) -> np.ndarray:
    surface_temp = ds['tmpsfc'].values
    surface_temp = surface_temp - 273.15 - 26
    return np.where(surface_temp > 0, surface_temp, 1e-6)


def moist_stability_parameter(ds: xr.Dataset) -> np.ndarray:
    potential_temp_surface = ds['hgtprs'].sel(lev=1000).values
    potential_temp_500 = ds['hgtprs'].sel(lev=500).values
    diff = potential_temp_500 - potential_temp_surface
    # return np.ones_like(diff, dtype=np.float64)
    return diff / 500.0 + 5.


def relative_humidity_parameter(ds: xr.Dataset) -> np.ndarray:
    rh_700_500 = ds['rhprs'].sel(lev=slice(700, 500)).values
    mean_rh = np.mean(rh_700_500, axis=0)
    rh = (mean_rh - 40.0) / 70.0

    return np.clip(rh, 0.0, 1.0)


def ocean_mask(ds: xr.Dataset) -> np.ndarray:
    lon = np.where(ds.lon < 180.0, ds.lon, 360.0 - ds.lon)
    yy, xx = np.meshgrid(lon, ds.lat)
    ocean_mask = globe.is_ocean(xx, yy)
    return np.where(ocean_mask, 1, 1e-6)


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
    thermal = thermal_parameter(ds)
    dynamic = dynamic_parameter(ds)
    return thermal * dynamic * ocean_mask(ds)
