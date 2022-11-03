from datetime import timedelta
from functools import reduce
import pandas as pd
from typing import Union, List


def _parse_tc_datetime(column: pd.Series):
    return pd.to_datetime(column, format='%Y-%m-%d %H:%M:%S')


def _group_observations_by_date(tc_labels: pd.DataFrame):
    def concat_values(values):
        return reduce(lambda agg, x: agg + [x], values, [])

    grouped = tc_labels.groupby('Date')

    tc_labels['TC'] = grouped['TC'].transform(
            lambda has_tc: reduce(lambda agg, x: agg and x, has_tc, True))

    concate_columns = [
            'TC Id',
            'First Observed',
            'Last Observed',
            'Latitude',
            'Longitude',
            'First Observed Type',
            'Will Develop to TC',
            'Developing Date',
        ]
    for col in concate_columns:
        tc_labels[col] = grouped[col].transform(concat_values)

    return tc_labels.drop_duplicates('Date', keep='first')


def filter_in_leadtime(tc: pd.DataFrame, leadtimes: Union[List[int], int] = None):
    if leadtimes is None:
        return tc

    if not isinstance(leadtimes, list):
        leadtimes = [leadtimes]

    # First, we will keep all negative cases.
    mask = ~tc['TC']

    # Then, loop through each lead time to get observations that belong to that leadtime.
    observation_dates = _parse_tc_datetime(tc['Date'])
    tc_first_observed_dates = _parse_tc_datetime(tc['First Observed'])
    for leadtime in leadtimes:
        leadtime = timedelta(hours=leadtime)
        mask |= (tc_first_observed_dates - observation_dates) == leadtime

    return tc[mask]


def load_label(label_path, group_observation_by_date=True, leadtime=None) -> pd.DataFrame:
    label = pd.read_csv(label_path, dtype={
        'TC Id': str,
        'First Observed': str,
        'Last Observed': str,
        'First Observed Type': str,
        'Will Develop to TC': str,
        'Developing Date': str,
    })

    label = filter_in_leadtime(label, leadtime)
    if group_observation_by_date:
        label = _group_observations_by_date(label)

    return label
