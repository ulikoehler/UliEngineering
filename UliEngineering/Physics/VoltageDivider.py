#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing different aspects and complexities of voltage dividers
"""
from UliEngineering.EngineerIO import normalizeEngineerInput, Quantity
from .Resistors import *


def unloadedVoltageDividerRatio(r1, r2) -> Quantity(""):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties or loading
    """
    r1, _ = autoNormalizeEngineerInput(r1)
    r2, _ = autoNormalizeEngineerInput(r2)
    return r1 / (r1 + r2)

def loadedVoltageDividerRatio(r1, r2, rl) -> Quantity(""):
    """
    Compute the denominator of the  division ratio of a voltage divider, not taking into account
    parasitic properties but loading.
    """
    r1, _ = autoNormalizeEngineerInput(r1)
    r2, _ = autoNormalizeEngineerInput(r2)
    rl, _ = autoNormalizeEngineerInput(rl)
    return r1 / (r1 + parallelResistors(r2, rl))


def computeTopResistor(rbottom, ratio) -> Quantity("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rbottom, _ = autoNormalizeEngineerInput(rbottom)
    ratio, _ = autoNormalizeEngineerInput(ratio)
    return -(rbottom * ratio) / (ratio - 1.0)


def computeBottomResistor(rtop, ratio) -> Quantity("Ω"):
    """
    Compute the bottom resistor of a voltage divider given the top resistor value
    and the division ration
    """
    rtop, _ = autoNormalizeEngineerInput(rtop)
    ratio, _ = autoNormalizeEngineerInput(ratio)
    return rtop * (1.0 / ratio - 1.0)
