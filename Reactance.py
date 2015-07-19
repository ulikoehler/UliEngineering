#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility to calculate idealized reactances.

Originally published at techoverflow.net
"""
import itertools
import math
from EngineerIO import *

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

if __name__ == "__main__":
    print(formatValue(capacitive_reactance("100 pF", "3.2e6"), "Ω"))