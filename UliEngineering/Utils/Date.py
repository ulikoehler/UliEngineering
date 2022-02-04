#!/usr/bin/env python3
from collections import namedtuple
from calendar import monthrange
import numpy as np
from datetime import datetime

__all__ = ["Date", "all_dates_in_year", "number_of_days_in_month",
    "generate_days", "generate_years", "generate_months",
    "extract_months", "extract_years", "extract_day_of_month",
    "extract_day_of_week", "is_first_day_of_month", "is_first_day_of_week",
    "is_month_change", "is_year_change"]

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

def extract_years(timestamps):
    """
    Given an 1D array of np.datetime64 timestamps,
    extract the month of each timestamps and return a
    numpy array of the same size, containing the year number
    (e.g. 2022)
    """
    return np.asarray([dt.year for dt in timestamps.astype(datetime)])

def extract_day_of_month(timestamps):
    """
    Given an 1D array of np.datetime64 timestamps,
    extract the month of each timestamps and return a
    numpy array of the same size, containing the day of month
    (1-31, depending on the month)
    """
    return np.asarray([dt.day for dt in timestamps.astype(datetime)])

def extract_day_of_week(timestamps):
    """
    Given an 1D array of np.datetime64 timestamps,
    extract the month of each timestamps and return a
    numpy array of the same size, containing the day of week
    (Monday=1, Sunday=7)
    """
    return np.asarray([dt.isoweekday() for dt in timestamps.astype(datetime)])

def is_first_day_of_month(timestamps):
    """
    Takes a Numpy array of np.datetime64.

    Returns a boolean array of the same length which is
    true if the given date is on the first day of any month.

    This is related to is_first_day_of_month(), but implements
    a slightly different algorithm
    """
    return extract_day_of_month(timestamps) == 1

def is_first_day_of_week(timestamps):
    """
    Takes a Numpy array of np.datetime64.

    Returns a boolean array of the same length which is
    true if the given date is on the first day of any week.
    """
    return extract_day_of_week(timestamps) == 1

def is_month_change(timestamps, first_value_is_change=False):
    """
    Takes a Numpy array of np.datetime64.

    Returns a boolean array of the same length which is
    true if the given date is the first date in the given array
    in that particular month

    If first_value_is_change is True, the first element of the array will be True,
    else it will be False.

    When using day-resolution datasets, this is often similar
    to using is_first_day_of_month(), however this function
    will only return True once for a given month,
    whereas is_first_day_of_month() will return True for ANY
    date that is on the 1st day of the month.
    """
    if len(timestamps) == 0:
        return np.asarray([], dtype=bool)
    return np.append([first_value_is_change],
        np.diff(extract_months(timestamps)).astype(bool))

def is_year_change(timestamps, first_value_is_change=False):
    """
    Takes a Numpy array of np.datetime64.

    If first_value_is_change is True, the first element of the array will be True,
    else it will be False.

    Returns a boolean array of the same length which is
    true if the given date is the first date in the given array
    in that particular year
    """
    if len(timestamps) == 0:
        return np.asarray([], dtype=bool)
    return np.append([first_value_is_change],
        np.diff(extract_years(timestamps)).astype(bool))

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
    # NOTE: This method is more efficient than the "string parsing"
    # method used by generate_months() and generate_years(),
    # but this only matters if generating a lot of entries
    # and it only works if the datetime64-represented
    # distance between units to generate is constant
    day_indexes = np.arange(ndays, dtype=np.int64) # 0, 1, ..., [ndays-1]
    startdate = np.datetime64(f'{year:02d}-{month:02d}-{day:02d}T00:00:00.000000', 'us')
    usec_per_day = int(1e6) * 86400 # 86.4k sec per day = 60*60*24s
    usec_offsets = day_indexes * usec_per_day
    return usec_offsets + startdate

def generate_months(nmonths, year=2022, month=1, day=1):
    """
    Generate an 1d array of [ndays] timestamps, starting at the given day,
    each timestamp being exactly one month from the previous one.
    The given date will be the first timestamp.

    Returns a array of np.datetime64[us]

    >>> generate_months(5, 2022, 1, 1)
    ['2022-01-01T00:00:00.000000',
     '2022-02-01T00:00:00.000000',
     '2022-03-01T00:00:00.000000',
     '2022-04-01T00:00:00.000000',
     '2022-05-01T00:00:00.000000']
    """
    return np.asarray([
        f'{year:04d}-{month+i:02d}-{day:02d}T00:00:00.000000'
        for i in range(nmonths)
    ], dtype='datetime64[us]')

def generate_years(nyears, year=2022, month=1, day=1):
    """
    Generate an 1d array of [ndays] timestamps, starting at the given day,
    each timestamp being exactly one year from the previous one.
    The given date will be the first timestamp.

    Returns a array of np.datetime64[us]

    >>> generate_years(5, 2022, 1, 1)
    ['2022-01-01T00:00:00.000000',
     '2023-01-01T00:00:00.000000',
     '2024-01-01T00:00:00.000000',
     '2025-01-01T00:00:00.000000',
     '2026-01-01T00:00:00.000000']
    """
    return np.asarray([
        f'{year+i:04d}-{month:02d}-{day:02d}T00:00:00.000000'
        for i in range(nyears)
    ], dtype='datetime64[us]')
