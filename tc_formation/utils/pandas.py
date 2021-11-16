import pandas as pd


class pandas_show():
    def __init__(self, all_rows=True, all_columns=True):
        self._all_rows = all_rows
        self._all_columns = all_columns

    def __enter__(self):
        pd.option_context(
            'display.min_rows', None,
            'display.max_rows', None,
            'display.max_columns', None,
            'display.max_colwidth', None)

    def __exit__(self):
        pd.reset_option('display.min_rows')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_colwidth')
        return True
