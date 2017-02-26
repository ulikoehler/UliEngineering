#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for processing and modifying datetime objects
"""
import datetime

__all__ = ["splice_date", "auto_strptime"]


def splice_date(datesrc, timesrc, tzinfo=None):
    """
    Create a new datetime that takes the date from datesrc and
    the time from timesrc. The tzinfo is taken from the tzinfo
    parameter. If it is None, it is taken from
    timesrc.tzinfo. No timezone conversion is performed.
    """
    tzinfo = timesrc.tzinfo if tzinfo is None else tzinfo
    return datetime.datetime(datesrc.year, datesrc.month, datesrc.day,
                             timesrc.hour, timesrc.minute, timesrc.second,
                             timesrc.microsecond, tzinfo=tzinfo)


def auto_strptime(s):
    """
    Parses a datetime in a number of formats,
    automatically recognizing which format is correct.

    Supported formats:
        %Y-%m-%d
        %Y-%m-%d %H
        %Y-%m-%d %H:%M
        %Y-%m-%d %H:%M:%S
        %Y-%m-%d %H:%M:%S.%f
        %H:%M:%S
        %H:%M:%S.%f
    """
    s = s.strip()
    ncolon = s.count(":")
    have_date = "-" in s
    if "." in s:  # Have fractional seconds
        dateformat = "%Y-%m-%d %H:%M:%S.%f" if have_date else "%H:%M:%S.%f"
    elif " " not in s:  # Have only date or have only time
        dateformat = "%Y-%m-%d" if have_date else "%H:%M:%S"
    elif ncolon == 0:  # Have date and time and no fractional but only hours
        dateformat = "%Y-%m-%d %H"
    elif ncolon == 1:  # Have date and time and no fractional but only h & m
        dateformat = "%Y-%m-%d %H:%M"
    else:  # Have date and time and no fractional but full time
        dateformat = "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.strptime(s, dateformat)
