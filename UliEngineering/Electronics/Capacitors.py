#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Area import normalize_area
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.EngineerIO.Length import normalize_length
from UliEngineering.Physics.Temperature import normalize_temperature_celsius

import numpy as np

__all__ = [
    "capacitor_lifetime",
    "capacitor_energy",
    "capacitor_charge",
    "parallel_plate_capacitors_capacitance",
    "capacitor_constant_current_charge_time",
    "capacitor_constant_current_discharge_time",
    "capacitor_voltage_by_energy",
    "capacitor_capacitance_by_energy",
    "capacitor_charging_energy",
]

@returns_unit("h")
def capacitor_lifetime(temp, nominal_lifetime="2000 h", nominal_lifetime_temperature="105 °C", A=10.):
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

@returns_unit("J")
@normalize_numeric_args
def capacitor_energy(capacitance, voltage):
    """
    Compute the total energy stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned as joules.
    """
    return 0.5 * capacitance * np.square(voltage)

@returns_unit("C")
@normalize_numeric_args
def capacitor_charge(capacitance, voltage):
    """
    Compute the total charge stored in a capacitor given:
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The charge is returned in coulombs.
    """
    return capacitance * voltage

@returns_unit("V")
@normalize_numeric_args
def capacitor_voltage_by_energy(capacitance, energy, starting_voltage="0V"):
    """
    Compute the voltage of a capacitor given:
    - The capacitance in farads
    - The energy stored in joules
    The voltage is returned in volts.
    """
    # Compute starting energy
    starting_energy = capacitor_energy(capacitance, starting_voltage)
    # Compute voltage
    return np.sqrt(2 * (energy + starting_energy) / capacitance)

@returns_unit("s")
@normalize_numeric_args
def capacitor_constant_current_discharge_time(capacitance, initial_voltage, current, target_voltage="0V"):
    """
    Compute the time it takes to charge a capacitor to [target_voltage]
    using a constant current.
    
    Keyword arguments:
    - capacitance: The capacitance of the capacitor in farads.
    - voltage: The initial voltage of the capacitor in volts.
    - current: The charge current in amperes.
    - target_voltage: The target voltage to discharge the capacitor to.
    
    Returns: The time in seconds.
    """
    # Use charge function with "negative current"
    # Since from the view of the charge function, its generating a negative
    # voltage charge, this will result in a positive time
    return capacitor_constant_current_charge_time(capacitance, target_voltage, current, initial_voltage)

@returns_unit("s")
@normalize_numeric_args
def capacitor_constant_current_charge_time(capacitance, target_voltage, current, initial_voltage="0V"):
    """
    Compute the time it takes to charge a capacitor to [target_voltage]
    using a constant current.
    
    Keyword arguments:
    - capacitance: The capacitance of the capacitor in farads.
    - initial_voltage: The initial voltage of the capacitor in volts.
    - current: The discharge current in amperes.
    - target_voltage: The target voltage to discharge the capacitor to.
    
    Returns: The time in seconds.
    """
    return capacitance * (initial_voltage - target_voltage) / current

@returns_unit("F")
def parallel_plate_capacitors_capacitance(area, distance, epsilon):
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
    area = normalize_area(area)
    distance = normalize_length(distance)
    epsilon = normalize_numeric(epsilon)
    return epsilon * area / distance

@returns_unit("F")
@normalize_numeric_args
def capacitor_capacitance_by_energy(energy, voltage, starting_voltage="0V"):
    """
    Compute the capacitance of a capacitor given:
    - The energy stored in joules
    - The voltage the capacitor is charged to
    - The starting voltage (optional, default 0V)
    
    The capacitance is returned in farads.
    
    The formula accounts for the energy difference between the final and starting voltages:
    Energy = 0.5 * C * (V_final^2 - V_starting^2)
    Therefore: C = 2 * Energy / (V_final^2 - V_starting^2)
    """
    voltage_squared_diff = np.square(voltage) - np.square(starting_voltage)
    return 2 * energy / voltage_squared_diff

@returns_unit("J")
@normalize_numeric_args
def capacitor_charging_energy(capacitance, end_voltage, starting_voltage="0V"):
    """
    Compute the energy required to charge a capacitor from a starting voltage to an end voltage.

    Parameters:
    - capacitance: The capacitance of the capacitor in farads.
    - end_voltage: The target voltage to charge the capacitor to in volts.
    - starting_voltage: The initial voltage of the capacitor in volts (default "0V").

    Returns:
    The energy required in joules.
    
    The energy required is the difference between the final and initial stored energy:
    Energy_required = 0.5 * C * (V_end^2 - V_start^2)
    """
    end_energy = capacitor_energy(capacitance, end_voltage)
    start_energy = capacitor_energy(capacitance, starting_voltage)
    return end_energy - start_energy
