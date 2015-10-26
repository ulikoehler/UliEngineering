#!/usr/bin/env python3
import scipy.constants
from .Resistors import *
from .EngineerIO import *
import math
import numpy as np

def capacitorEnergy(capacitance, voltage):
    """
    Compute the total energy stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned as joules.

    >>> capacitorEnergy("1.5 F", "5.0 V")
    18.75
    >>> capacitorEnergy("1.5 F", "0.0 V")
    0.0
    """
    capacitance, _ = normalizeEngineerInputIfStr(capacitance)
    voltage, _ = normalizeEngineerInputIfStr(voltage)
    return 0.5 * capacitance * np.square(voltage)

def capacitorCharge(capacitance, voltage):
    """
    Compute the total charge stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned in coulombs.

    >>> capacitorCharge("1.5 F", "5.0 V")
    7.5
    >>> capacitorCharge("1.5 F", "0.0 V")
    0.0
    """
    capacitance, _ = normalizeEngineerInputIfStr(capacitance)
    voltage, _ = normalizeEngineerInputIfStr(voltage)
    return capacitance * voltage

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Usage example
    print(formatValue(capacitorEnergy("100 mF", "1.2 V"), "J"))
