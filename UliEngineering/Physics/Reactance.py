#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to calculate idealized reactances.

Originally published at techoverflow.net
"""
import itertools
import numpy as np
from UliEngineering.EngineerIO import autoNormalizeEngineerInput, Quantity

def capacitive_reactance(c, f=1000.0) -> Quantity("Ω"):
    """
    Compute the capacitive reactance for a given capacitance and frequency.
    """
    c, _ = autoNormalizeEngineerInput(c)
    f, _ = autoNormalizeEngineerInput(f)
    return 1.0 / (2 * np.pi * f * c)

def inductive_reactance(l, f=1000.0) -> Quantity("Ω"):
    """
    Compute the inductive reactance for a given inductance and frequency.
    """
    l, _ = autoNormalizeEngineerInput(l)
    f, _ = autoNormalizeEngineerInput(f)
    return 2 * np.pi * f * l
