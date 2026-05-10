#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for computations related to noise density."""
from typing import Annotated

import numpy as np
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Frequency import FrequencyHz, normalize_frequency
from ._normalize import normalize_with_known_units

__all__ = ["actual_noise", "noise_density",
           "normalize_voltage", "VoltageVolt",
           "normalize_frequency", "FrequencyHz"]

def normalize_voltage(voltage: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(voltage, {"V": 1.0, "Volt": 1.0, "volt": 1.0, "mV": 1e-3, "µV": 1e-6, "nV": 1e-9, "pV": 1e-12, "kV": 1e3, "MV": 1e6}, quantity_name="voltage")

VoltageVolt = Annotated[NormalizedComputable, normalize_voltage]

@returns_unit("V")
def actual_noise(density: VoltageVolt, bandwith: FrequencyHz):
    """
    Compute the actual noise given a noise density in x/√Hz and a bandwith in ΔHz.

    >>> autoFormat(actualNoise, "100 µV", "100 Hz")
    '1.00 mV'
    """
    density = normalize_voltage(density)
    bandwith = normalize_frequency(bandwith)
    return np.sqrt(bandwith) * density

@returns_unit("V/√Hz")
def noise_density(actual_noise: VoltageVolt, bandwith: FrequencyHz):
    """
    Compute the noise density given:
     - A noise density in x/√Hz where x is any unit
     - A bandwith in ΔHz

    >>> formatValue(noiseDensity("1.0 mV", "100 Hz"), "V/√Hz")
    '100 μV/√Hz'
    """
    actual_noise = normalize_voltage(actual_noise)
    bandwith = normalize_frequency(bandwith)
    return actual_noise / np.sqrt(bandwith)
