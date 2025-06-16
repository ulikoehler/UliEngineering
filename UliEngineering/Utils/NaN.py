#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections.abc import Iterable
import numpy as np

from UliEngineering.EngineerIO.Types import NormalizeResult

def none_to_nan(value):
    """
    Convert None to NaN, otherwise return the value unchanged.
    This is useful for normalizing values in arrays.
    """
    # NOTE: string is iterable, so we need to check for that first
    if isinstance(value, str):
        return np.nan if value.strip() == '' else value.strip()
    # NOTE: NormalizeResult is a namedtuple i.e. iterable, so we need to handle it separately
    if isinstance(value, NormalizeResult):
        return NormalizeResult(
            prefix=value.prefix,
            value=none_to_nan(value.value),
            unit_prefix=value.unit_prefix,
            unit=value.unit
        )
    if isinstance(value, Iterable):
        # If it's an iterable, convert each element
        return [none_to_nan(elem) for elem in value]
    # Else: Assume a simple value
    if value is None:
        return np.nan
    return value
