#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
import itertools
import math
from .EngineerIO import *
from .Resistors import *

def unloadedVoltageDividerRatio(r1, r2):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties or loading

    >>> unloadedVoltageDividerRatio(1000.0, 1000.0)
    0.5
    """
    if isinstance(r1, str):
        r1, _ = normalizeEngineerInput(r1)
    if isinstance(r2, str):
        r2, _ = normalizeEngineerInput(r2)
    return r1 / (r1 + r2)

def loadedVoltageDividerRatio(r1, r2, rl):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties but loading.

    >>> loadedVoltageDividerRatio(1000.0, 1000.0, 1e60)
    0.5
    >>> loadedVoltageDividerRatio(1000.0, 1000.0, 1000.0)
    0.6666666666666666
    """
    if isinstance(r1, str):
        r1, _ = normalizeEngineerInput(r1)
    if isinstance(r2, str):
        r2, _ = normalizeEngineerInput(r2)
    if isinstance(rl, str):
        rl, _ = normalizeEngineerInput(rl)
    return r1 / (r1 + parallelResistors(r2, rl))


def computeTopResistor(rbottom, ratio):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration

    >>> computeTopResistor(1000.0, 0.5)
    1000.0
    """
    if isinstance(rbottom, str):
        rbottom, _ = normalizeEngineerInput(rbottom)
    if isinstance(ratio, str):
        ratio, _ = normalizeEngineerInput(ratio)
    return -(rbottom * ratio) / (ratio - 1)


def computeBottomResistor(rtop, ratio):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration

    >>> computeBottomResistor(1000.0, 0.5)
    1000.0
    """
    if isinstance(rtop, str):
        rtop, _ = normalizeEngineerInput(rtop)
    if isinstance(ratio, str):
        ratio, _ = normalizeEngineerInput(ratio)
    return rtop * (1 / ratio - 1)

# Usage example: Find and print the E48 resistor closest to 5 kOhm
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(loadedVoltageDividerRatio("1kΩ", "1kΩ", "10 MΩ"))
