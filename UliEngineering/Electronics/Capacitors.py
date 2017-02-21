#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np


def capacitor_energy(capacitance, voltage) -> Unit("J"):
    """
    Compute the total energy stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned as joules.
    """
    capacitance = normalize_numeric(capacitance)
    voltage = normalize_numeric(voltage)
    return 0.5 * capacitance * np.square(voltage)


def capacitor_charge(capacitance, voltage) -> Unit("C"):
    """
    Compute the total charge stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The charge is returned in coulombs.
    """
    capacitance = normalize_numeric(capacitance)
    voltage = normalize_numeric(voltage)
    return capacitance * voltage
