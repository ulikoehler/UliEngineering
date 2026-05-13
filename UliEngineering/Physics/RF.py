#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for computations related to noise density."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Frequency import FrequencyHz, normalize_frequency
from ._normalize import normalize_with_known_units
import numpy as np

__all__ = [
     'quality_factor', 'resonant_impedance', 'resonant_frequency',
     'resonant_inductance',
     'normalize_inductance', 'InductanceHenry',
     'normalize_capacitance', 'CapacitanceFarad',
     'normalize_frequency', 'FrequencyHz']

def normalize_inductance(inductance: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(inductance, {"H": 1.0, "Henry": 1.0, "henry": 1.0, "mH": 1e-3, "µH": 1e-6, "nH": 1e-9, "pH": 1e-12, "kH": 1e3}, quantity_name="inductance")

def normalize_capacitance(capacitance: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(capacitance, {"F": 1.0, "Farad": 1.0, "farad": 1.0, "mF": 1e-3, "µF": 1e-6, "nF": 1e-9, "pF": 1e-12, "kF": 1e3}, quantity_name="capacitance")

InductanceHenry = Annotated[NormalizedComputable, normalize_inductance]
CapacitanceFarad = Annotated[NormalizedComputable, normalize_capacitance]

@returns_unit("")
def quality_factor(frequency: FrequencyHz, bandwidth: FrequencyHz):
    """
    Compute the quality factor of a resonant circuit from the frequency and the bandwidth.

    Q = frequency / bandwidth.

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> quality_factor("8.000 MHz", "1 kHz")
    8000.0
    """
    frequency = normalize_frequency(frequency)
    bandwidth = normalize_frequency(bandwidth)
    return frequency / bandwidth

@returns_unit("Ω")
def resonant_impedance(L: InductanceHenry, C: CapacitanceFarad, Q=100.):
    """
    Compute the resonant impedance of a resonant circuit.

    R_res = sqrt(L / C) / Q.

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_impedance("100 uH", "10 nF", Q=30.0)
    3.333333333333333
    >>> auto_format(resonant_impedance, "100 uH", "10 nF", Q=30.0)
    '3.33 Ω'
    """
    L = normalize_inductance(L)
    C = normalize_capacitance(C)
    return np.sqrt(L / C) / Q

@returns_unit("Hz")
def resonant_frequency(L: InductanceHenry, C: CapacitanceFarad):
    """
    Compute the resonant frequency of a resonant circuit given the inductance and capacitance.

    f = 1 / (2 * pi * sqrt(L * C)).

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_frequency("100 uH", "10 nF")
    159154.94309189534
    >>> auto_format(resonant_frequency, "100 uH", "10 nF")
    '159 kHz'
    """
    L = normalize_inductance(L)
    C = normalize_capacitance(C)
    return 1 / (2 * np.pi * np.sqrt(L * C))

@returns_unit("H")
def resonant_inductance(fres: FrequencyHz, C: CapacitanceFarad):
    """
    Compute the inductance of a resonant circuit given the resonant frequency and its capacitance.

    L = 1 / (4 * pi² * fres² * C).

    Source: http://www.c-max-time.com/tech/antenna.php

    >>> resonant_inductance("250 kHz", "10 nF")
    4.052847345693511e-05
    >>> auto_format(resonant_inductance, "250 kHz", "10 nF")
    '40.5 µH'
    """
    fres = normalize_frequency(fres)
    C = normalize_capacitance(C)
    return 1 / (4 * np.pi**2 * fres**2 * C)
