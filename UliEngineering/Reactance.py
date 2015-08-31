#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility to calculate idealized reactances.

Originally published at techoverflow.net
"""
import itertools
import math
from EngineerIO import *
from sympy import *

__author__ = "Uli Koehler"
__license__ = "CC0 1.0 Universal"
__version__ = "1.0"

def capacitive_reactance(c, f=1000.0):
    """
    Compute the capacitive reactance for a given capacitance and frequency.
    """
    if isinstance(c, str):
        c, _ = normalizeEngineerInput(c)
    if isinstance(f, str):
        f, _ = normalizeEngineerInput(f)
    return 1.0/(2*math.pi*f*c)

def inductive_reactance(l, f=1000.0):
    """
    Compute the inductive reactance for a given inductance and frequency.
    """
    if isinstance(l, str):
        l, _ = normalizeEngineerInput(l)
    if isinstance(f, str):
        f, _ = normalizeEngineerInput(f)
    return 2*math.pi*f*l

if __name__ == "__main__":
    print(formatValue(capacitive_reactance("100 pF", "3.2e6"), "Ω"))
    print(formatValue(inductive_reactance("100 µH", "3.2e6"), "Ω"))