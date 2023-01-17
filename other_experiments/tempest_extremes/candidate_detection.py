# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

from datetime import datetime
import pandas as pd

# # Checking the output of TempestExtremes

# First, let's specify the .netCDF file we want to use.

nc_date = datetime(2008, 5, 6, 18, 0)
nc_file_name = f'fnl_{nc_date.strftime("%Y%m%d_%H_%M")}.nc'
nc_file_path = f'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_PRESsfc_100_260/6h/{nc_file_name}'

# Then, execute TempestExtremes's CLI to detect storms candidates.

file_date = nc_date.strftime('%Y-%m-%d')
file_time = nc_date.strftime('%H:%M:00')
print(file_date, file_time)
# !module load nco &> /dev/null && ncecat -u time "$nc_file_path" -O tmp.nc && ncdump -t -v time tmp.nc
# #!module load cdo &> /dev/null && cdo -setreftime,"$file_date","$file_time" tmp.nc test.nc
# !module load cdo &> /dev/null && cdo -setreftime,'2008-05-06','18:00:00' -settaxis,'2008-05-06','18:00:00' tmp.nc test.nc
# !module load nco &> /dev/null && ncdump -t -v time test.nc
# !DetectNodes \
#     --in_data ./test.nc \
#     --out test_tempest_out.txt \
#     --searchbymin pressfc \
#     --closedcontourcmd "pressfc,100.0,5.5,0" \
#     --mergedist 6.0 \
#     --outputcmd "pressfc,min,0" \
#     --regional
# !cat test_tempest_out.txt

# Load the output of Tempest Extremes to pandas dataframe.

df: pd.DataFrame = pd.read_csv('test_tempest_out.txt',
                               sep='\t',
                               skiprows=1,
                               names=['Lon (pixels)', 'Lat (pixels)', 'Lon', 'Lat'])
df.head(10)

# We will then look at the pressure surface of file
# to see if Tempest Extremes does a good job.

# +
import matplotlib.pyplot as plt # noqa
import numpy as np # noqa
from tc_formation.plots import decorators, observations as plt_obs # noqa
import xarray as xr # noqa


@decorators._with_axes
@decorators._with_basemap
def plot_points(
        dataset: xr.Dataset,
        points: np.ndarray,
        ax: plt.Axes = None,
        *args, **kwargs):
    ax.plot(points[:, 1], points[:, 0], 'go')

ds = xr.load_dataset(nc_file_path)
fig, ax = plt.subplots(figsize=(18, 12))
plt_obs.plot_variablef(dataset=ds, variable='pressfc', ax=ax)
plt_obs.plot_wind(dataset=ds, pressure_level=700, skip=4, ax=ax)
plot_points(dataset=ds, points=df[['Lat', 'Lon']].values, ax=ax)
fig.tight_layout()
# -

# Now, we will move on with merging candidates' path.
# In order to do this, we have to create a list of TC files (say 5),
# put them into a list, then run `DetectNodes` cli.
# The output of that utility will be used by `StitchNodes` cli.

# +
from datetime import timedelta # noqa
import subprocess # noqa

nc_start_date = datetime(2008, 5, 6, 18, 0)
num_files = 20

for file_num in range(num_files):
    date = nc_start_date + timedelta(hours=file_num * 6)
    file_name = f'fnl_{date.strftime("%Y%m%d_%H_%M")}.nc'
    file_path = f'data/nolabels_wp_ep_alllevels_ABSV_CAPE_RH_TMP_HGT_VVEL_UGRD_VGRD_PRESsfc_100_260/6h/{file_name}'
    file_date = date.strftime('%Y-%m-%d')
    file_time = date.strftime('%H:%M:00')

    add_time_cmd = f'module load nco && ncecat -u time {file_path} -O tmp_{file_num}.nc'
    set_time_cmd = f'module load cdo && cdo -setreftime,{file_date},{file_time} -settaxis,{file_date},{file_time} tmp_{file_num}.nc test_{file_num}.nc'

    subprocess.call(add_time_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    subprocess.call(set_time_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)


# Output these nc files to a txt file.
with open('test_nc_files.txt', 'w') as outfile:
    outfile.writelines([f'test_{i}.nc\n' for i in range(num_files)])

# Call DetectNodes
# !DetectNodes \
#     --in_data_list ./test_nc_files.txt \
#     --out ./test_tempest_out_list.txt \
#     --searchbymin pressfc \
#     --closedcontourcmd "pressfc,100.0,5.5,0" \
#     --mergedist 6.0 \
#     --regional
# !cat test_tempest_out_list.txt
# -

# !StitchNodes \
#     --in_list test_tempest_out_list.txt \
#     --out test_tempest_out_track.txt \
#     --in_fmt "lon,lat" \
#     --range 8.0 --mintime "54h" \
#     --maxgap "24h" \
#     --threshold "lat,<=,50.0,10;lat,>=,-50.0,10"

# After getting the result of `StitchNodes` cli,
# we will have to parse its output.

# +
from collections import namedtuple # noqa
from datetime import datetime # noqa


class TropicalCycloneCandidate:
    CandidateTrack = namedtuple('CandidateTrack', ['lat_pixels', 'lon_pixels', 'lat', 'lon', 'year', 'month', 'day', 'hour'])

    @property
    def first_observed_date(self):
        return self._first_observed_date

    @property
    def locations(self):
        return self._tracks[['lat', 'lon']].values

    def __init__(self, tracks: list[CandidateTrack]):
        first_observed = tracks[0]
        self._first_observed_date = datetime(
            first_observed.year, first_observed.month, first_observed.day, first_observed.hour, 0)
        self._tracks = pd.DataFrame([track._asdict() for track in tracks])

    @staticmethod
    def create_track(lat_pixels: int, lon_pixels: int, lat: float, lon: float, year: int, month: int, day: int, hour: int):
        return TropicalCycloneCandidate.CandidateTrack(
            lat_pixels=lat_pixels,
            lon_pixels=lon_pixels,
            lat=lat,
            lon=lon,
            year=year,
            month=month,
            day=day,
            hour=hour,
        )


def parse_stitch_nodes_output(outputfile: str) -> list[TropicalCycloneCandidate]:
    candidates = []

    with open(outputfile, 'r') as file:
        while True:
            try:
                line = next(file)
            except StopIteration:
                # We've reached the end of file.
                break

            assert line.startswith('start'), 'Wrong file or format!'
            _, nb_tracks, *_ = line.split('\t')
            track_lines = [next(file) for _ in range(int(nb_tracks))]

            candidate_track = []
            for track_line in track_lines:
                _, lon_pixels, lat_pixels, lon, lat, year, month, day, hour = track_line.split('\t')
                candidate_track.append(TropicalCycloneCandidate.create_track(
                    lat_pixels=int(lat_pixels),
                    lon_pixels=int(lon_pixels),
                    lat=float(lat),
                    lon=float(lon),
                    year=int(year),
                    month=int(month),
                    day=int(day),
                    hour=int(hour),
                ))

            candidates.append(TropicalCycloneCandidate(candidate_track))

    return candidates

candidates = parse_stitch_nodes_output('./test_candidate_detection.txt')
for c in candidates[:2]:
    print(c.first_observed_date, c.locations)
# -

# Now, we will plot to see what do we get from the StitchNodes cli.

# +
@decorators._with_axes
@decorators._with_basemap
def plot_candidate_track(
        dataset: xr.Dataset,
        candidates: list[TropicalCycloneCandidate],
        ax: plt.Axes = None,
        *args, **kwargs):
    for i, candidate in enumerate(candidates[:50]):
        tracks = candidate.locations
        ax.plot(tracks[:, 1], tracks[:, 0], 'o-', label=str(i))


ds = xr.load_dataset(nc_file_path)
fig, ax = plt.subplots(figsize=(18, 12))
plt_obs.plot_variablef(dataset=ds, variable='pressfc', ax=ax)
# plt_obs.plot_wind(dataset=ds, pressure_level=700, skip=4, ax=ax)
plot_candidate_track(dataset=ds, candidates=candidates, ax=ax)
ax.legend()
fig.tight_layout()
