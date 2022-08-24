#!/bin/env python3

import argparse
from datetime import datetime
import os
import pandas as pd


def parse_arguments(args=None):
    parser = argparse.ArgumentParser('TC Train Test Split')

    parser.add_argument(
        '--test-from', '-tf',
        dest='test_start',
        action='store',
        required=True,
        help='Test data start from this date. Format: YYYYMMDD.')
    parser.add_argument(
        '--val-from', '-vf',
        dest='val_start',
        action='store',
        help='(Optional) Validation data start from this date. Format: YYYYMMDD.')

    parser.add_argument(
            '--labels',
            action='store',
            required=True,
            help='Path to csv file to split dataset.')

    return parser.parse_args(args)

def parse_date(datestr: str):
    return datetime.strptime(datestr, '%Y%m%d')

if __name__ == '__main__':
    args = parse_arguments()

    # Make sure that validation start date is before than the test data.
    if args.val_start is not None:
        assert args.val_start < args.test_start, f'ERR: Validation date {args.val_start} must be before test date {args.test_start}'

    tc = pd.read_csv(args.labels)
    tc['Date'] = pd.to_datetime(tc['Date'], format='%Y-%m-%d %H:%M:%S')

    # Obtain test labels.
    test_date = parse_date(args.test_start)
    test_labels = tc[tc['Date'] >= test_date]

    # Obtain val and train labels.
    if args.val_start:
        val_date = parse_date(args.val_start)
        val_labels = tc[(tc['Date'] >= val_date) & (tc['Date'] < test_date)]
        train_labels = tc[tc['Date'] < val_date]
    else:
        val_labels = None
        train_labels = tc[tc['Date'] < test_date]

    # Finally, output to files.
    labels = [
        [train_labels, 'train'],
        [val_labels, 'val'],
        [test_labels, 'test'],
    ]
    for label, label_type in labels:
        if label is not None:
            path = args.labels.replace('.csv', f'_{label_type}.csv')
            label.to_csv(path, index=False)
