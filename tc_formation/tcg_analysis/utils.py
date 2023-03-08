from datetime import datetime
import os


def parse_date(path: str):
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    datepart = '_'.join(filename.split('_')[1:])
    return datetime.strptime(datepart, '%Y%m%d_%H_%M')
