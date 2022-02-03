#!/usr/bin/env python3
from collections import namedtuple
from calendar import monthrange
import numpy as np
from datetime import datetime

__all__ = ["Date", "all_dates_in_year", "number_of_days_in_month",
    "generate_days", "extract_months"]

Date = namedtuple("Date", ["year", "month", "day"])

def number_of_days_in_month(year=2019, month=1):
    """
    Returns the number of days in a month, e.g. 31 in January (month=1).
    Takes into account leap days.
    """
    return monthrange(year, month)[1]

def all_dates_in_year(year=2019):
    """
    Iterates all dates in a specific year, taking into account leap days.
    Yields Date() objects.
    """
    for month in range(1, 13): # Month is always 1..12
        for day in range(1, number_of_days_in_month(year, month) + 1):
            yield Date(year, month, day)

def extract_months(timestamps):
    """
    Given an 1D array of np.datetime64 timestamps,
    extract the month of each timestamps and return a
    numpy array of the same size, containing the month number
    (january = 1)
    """
    return np.asarray([dt.month for dt in timestamps.astype(datetime)])

def generate_days(ndays, year=2022, month=1, day=1):
    """
    Generate an 1d array of [ndays] timestamps, starting at the given day,
    each timestamp being exactly one day from the previous one.
    The given date will be the first timestamp.

    Returns a array of np.datetime64[us]

    >>> generate_days(5, 2022, 1, 1)
    ['2022-01-01T00:00:00.000000',
     '2022-01-02T00:00:00.000000',
     '2022-01-03T00:00:00.000000',
     '2022-01-04T00:00:00.000000',
     '2022-01-05T00:00:00.000000']
    """
    day_indexes = np.arange(ndays, dtype=np.int64) # 0, 1, ..., [ndays-1]
    startdate = np.datetime64(f'{year:02d}-{month:02d}-{day:02d}T00:00:00.000000', 'us')
    usec_per_day = int(1e6) * 86400 # 86.4k sec per day = 60*60*24s
    usec_offsets = day_indexes * usec_per_day
    return usec_offsets + startdate