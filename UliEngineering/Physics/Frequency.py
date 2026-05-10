#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for frequencies."""
from typing import Annotated

from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric

from ._normalize import normalize_with_known_units

__all__ = ["frequency_to_period", "period_to_frequency", "normalize_frequency", "normalize_rpm",
           "FrequencyHz", "RotationFrequency", "RotationRate"]


def normalize_rpm(speed: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(speed, {"rpm": 1.0 / 60.0}, quantity_name="rotational speed")


def normalize_frequency(speed: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(speed, {"rpm": 1.0 / 60.0, "Hz": 1.0}, quantity_name="frequency")


# Unit type annotations
FrequencyHz = Annotated[NormalizedComputable, normalize_frequency]
RotationFrequency = FrequencyHz  # Alias for consistency
RotationRate = Annotated[NormalizedComputable, normalize_rpm]

@returns_unit("s")
def frequency_to_period(frequency: NormalizableArgument):
    """
    Compute the period associated with a frequency.

    Parameters
    ----------
    frequency : number or Engineer string or NumPy array-like
        The frequency in Hz

    """
    frequency = normalize_numeric(frequency) if isinstance(frequency, str) else frequency
    return 1./frequency

@returns_unit("Hz")
def period_to_frequency(period: NormalizableArgument):
    """
    Compute the frequency associated with a period.

    Parameters
    ----------
    period : number or Engineer string or NumPy array-like
        The period in seconds
    """
    period = normalize_numeric(period) if isinstance(period, str) else period
    return 1./period
