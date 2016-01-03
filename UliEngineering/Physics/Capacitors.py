#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.constants
from UliEngineering.EngineerIO import normalizeEngineerInputIfStr, formatValue
import numpy as np

def capacitor_energy(capacitance, voltage):
    """
    Compute the total energy stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned as joules.
    """
    capacitance, _ = normalizeEngineerInputIfStr(capacitance)
    voltage, _ = normalizeEngineerInputIfStr(voltage)
    return 0.5 * capacitance * np.square(voltage)

def capacitor_charge(capacitance, voltage):
    """
    Compute the total charge stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The charge is returned in coulombs.
    """
    capacitance, _ = normalizeEngineerInputIfStr(capacitance)
    voltage, _ = normalizeEngineerInputIfStr(voltage)
    return capacitance * voltage

