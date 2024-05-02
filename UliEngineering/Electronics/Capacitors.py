#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
from UliEngineering.Physics.Temperature import normalize_temperature_celsius

import numpy as np

__all__ = [
    "capacitor_lifetime",
    "capacitor_energy",
    "capacitor_charge",
    "parallel_plate_capacitors_capacitance"
]

def capacitor_lifetime(temp, nominal_lifetime="2000 h", nominal_lifetime_temperature="105 °C", A=10.) -> Unit("h"):
    """
    Estimate the lifetime of a capacitor given
    * Its working temperature (i.e. internal temperature)
    * Its nominal lifetime at a nominal lifetime temperature
    * Coefficient A: Temperature difference for which to assume a halving of the lifetime

    Based on:
    https://www.illinoiscapacitor.com/tech-center/life-calculators.aspx
    """
    temp = normalize_temperature_celsius(temp)
    nominal_lifetime_temperature = normalize_temperature_celsius(nominal_lifetime_temperature)
    nominal_lifetime = normalize_numeric(nominal_lifetime)
    # Compute lifetime
    tdelta = temp - nominal_lifetime_temperature
    return nominal_lifetime * 2**(-(tdelta/A))


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

def parallel_plate_capacitors_capacitance(area, distance, epsilon) -> Unit("F"):
    """
    Compute the capacitance of two parallel plate capacitors in parallel
    given the area, distance, and permittivity of the dielectric.

    Parameters:
    - area: The area of the capacitor plates in square meters.
    - distance: The distance between the capacitor plates in meters.
    - epsilon: The permittivity of the dielectric material between the capacitor plates.

    Returns:
    The capacitance of the parallel plate capacitors in farads (F).
    """
    area = normalize_numeric(area)
    distance = normalize_numeric(distance)
    epsilon = normalize_numeric(epsilon)
    return epsilon * area / distance