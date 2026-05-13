#!/usr/bin/env python3
import numpy as np

__all__ = [
    "yearly_interest_to_equivalent_monthly_interest",
    "yearly_interest_to_equivalent_daily_interest",
    "yearly_interest_to_equivalent_arbitrary_interest",
    "interest_apply_multiple_times",
    "extrapolate_interest_to_timestamps"
]

def yearly_interest_to_equivalent_monthly_interest(interest):
    """
    Given a yearly interest such as 0.022 (= 2.2%), or a numpy ndarray of interests.
    
    computes the equivalent monthly interest so that the following holds True:

    (1+monthly_interest)**12-1 == yearly_interest.

    Parameters
    ----------
    interest : float or numpy.ndarray
        Yearly interest rate.

    Returns
    -------
    float or numpy.ndarray
        Equivalent monthly interest rate.
    
    """
    # 12 months per year
    # monthly interest is 12th root of yearly interest
    # https://techoverflow.net/2022/02/02/numpy-nth-root-how-to/
    return np.power(1.+interest, (1/12.))-1.

def yearly_interest_to_equivalent_daily_interest(interest, days_per_year=365.25):
    """
    Given a yearly interest such as 0.022 (= 2.2%), or a numpy ndarray of interests.
    
    computes the equivalent daily interest so that the following holds True:

    (1+daily_interest)**365.25-1 == yearly_interest.

    It is highly recommended to use the Julian year (exactly 365.25 days/year),
    as using any other value like 365 will mean that over time spans containing leap
    years, the daily interest will not be equivalent to the yearly interest.

    Parameters
    ----------
    interest : float or numpy.ndarray
        Yearly interest rate.
    days_per_year : float, optional
        Number of days per year. Default is 365.25 (Julian year).

    Returns
    -------
    float or numpy.ndarray
        Equivalent daily interest rate.
    
    """
    # https://techoverflow.net/2022/02/02/numpy-nth-root-how-to/
    return np.power(1.+interest, (1/days_per_year))-1.

def yearly_interest_to_equivalent_arbitrary_interest(interest, seconds=1):
    """
    Given a yearly interest such as 0.022 (= 2.2%), or a numpy ndarray of interests.
    
    computes the equivalent interest rate for a timespan of [seconds] so that the total.
    interest of applying the rate for that timespan equals the yearly interest.

    This function is using the Julian year (exactly 365.25 days/year) as a reference
    for how many seconds a year is, as using any other value like 365 will mean that over
    time spans containing leap years, the daily interest will not be equivalent to the
    yearly interest.

    Parameters
    ----------
    interest : float or numpy.ndarray
        Yearly interest rate.
    seconds : float, optional
        Timespan in seconds. Default is 1.

    Returns
    -------
    float or numpy.ndarray
        Equivalent interest rate for the given timespan.
    
    """
    # https://techoverflow.net/2022/02/02/numpy-nth-root-how-to/
    # scipy.constants.Julian_year == 31557600.0
    return np.power(1.+interest, (seconds/31557600.0))-1.

def interest_apply_multiple_times(interest, times):
    """
    Given a yearly interest such as 0.022 (= 2.2%), apply it to the given values.

    For example, interest_apply_multiple_times(0.022, 5) will return the equivalent.
    interest of having an interest of 2.2% 5 years in a row (including compound interest).

    times may be a floating point number.

    Computes (1+interest)**times - 1.

    Parameters
    ----------
    interest : float or numpy.ndarray
        Interest rate.
    times : float
        Number of times to apply the interest.

    Returns
    -------
    float or numpy.ndarray
        Total interest after applying the interest multiple times.
    
    """
    return np.power(1.+interest, times)-1.

def extrapolate_interest_to_timestamps(interest, timestamps):
    """
    Given a yearly interest such as 0.022 (= 2.2%), extrapolate the total interest.
    
    factor up to each time point X in the timestamps array. The first timestamp in the.
    array is assumed to be 1.0.

    This works by computing the time difference for each timestamps t_i to the first
    timestamp t_0 and then applying the correct exponent to the interest to obtain a
    yearly equivalent interest factor.

    This function returns a numpy array of interest factors (typically >1), not
    interests(typically <1) for easy multiplication onto any values.

    Parameters
    ----------
    interest : float
        Yearly interest rate.
    timestamps : numpy.ndarray
        Array of datetime64 timestamps.

    Returns
    -------
    numpy.ndarray
        Array of interest factors for each timestamp.
    
    """
    # We compute in microseconds, not nanoseconds!
    timestamps = timestamps.astype('datetime64[us]')
    tdelta_us = (timestamps - timestamps[0]).astype(np.int64)
    # scipy.constants.Julian_year == 31557600.0,
    # multiplied by 1e6 due to microseconds timestamps
    return np.power(1.+interest, (tdelta_us/31557600e6))