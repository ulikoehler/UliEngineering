#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Annotated

from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Area import normalize_area
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Length import normalize_length
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics._normalize import normalize_with_known_units
from UliEngineering.Physics.Temperature import normalize_temperature
from UliEngineering.Electronics.Diode import normalize_diode_model, normalize_current, CurrentA, normalize_voltage, VoltageV

import numpy as np

__all__ = [
    "capacitor_lifetime",
    "capacitor_energy",
    "capacitor_charge",
    "capacitor_rc_time_constant",
    "parallel_plate_capacitors_capacitance",
    "capacitor_constant_current_charge_time",
    "capacitor_constant_current_discharge_time",
    "capacitor_resistor_charge_time",
    "capacitor_resistor_discharge_time",
    "capacitor_voltage_by_energy",
    "capacitor_capacitance_by_energy",
    "capacitor_charging_energy",
    "normalize_capacitance", "CapacitanceFarad",
    "normalize_resistance", "ResistanceOhm",
    "normalize_energy", "EnergyJ",
    "normalize_permittivity", "PermittivityFm",
]


def normalize_capacitance(C: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(C, {"F": 1.0, "µF": 1e-6, "nF": 1e-9, "pF": 1e-12, "mF": 1e-3, "uF": 1e-6}, quantity_name="capacitance")

def normalize_resistance(R: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(R, {"Ω": 1.0, "ohm": 1.0, "kΩ": 1e3, "MΩ": 1e6, "mΩ": 1e-3}, quantity_name="resistance")

def normalize_energy(E: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(E, {"J": 1.0, "mJ": 1e-3, "µJ": 1e-6, "kJ": 1e3}, quantity_name="energy")

def normalize_permittivity(epsilon: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(epsilon, {"F/m": 1.0, "F/meter": 1.0, "F/cm": 100.0}, quantity_name="permittivity")

CapacitanceFarad = Annotated[NormalizedComputable, normalize_capacitance]
ResistanceOhm = Annotated[NormalizedComputable, normalize_resistance]
EnergyJ = Annotated[NormalizedComputable, normalize_energy]
PermittivityFm = Annotated[NormalizedComputable, normalize_permittivity]
def _capacitor_resistor_model_time(capacitance, resistance, initial_drive_voltage, target_drive_voltage, diode_model, initial_voltage, target_voltage):
    initial_integral = diode_model.series_current_integral(initial_drive_voltage, resistance)
    target_integral = diode_model.series_current_integral(target_drive_voltage, resistance)
    time = capacitance * (initial_integral - target_integral)
    same_voltage = np.equal(initial_voltage, target_voltage)
    if np.isscalar(time):
        return 0.0 if same_voltage else time
    return np.where(same_voltage, 0.0, time)


@returns_unit("s")
def capacitor_rc_time_constant(capacitance: CapacitanceFarad, resistance: ResistanceOhm):
    """
    Compute the R/C time constant tau = R * C of a resistor-capacitor network.

    Parameters.
    ----------
    capacitance : number or Engineer string
        The capacitance in farads.
    resistance : number or Engineer string
        The resistance in ohms.

    Returns
    -------
    float
        The time constant in seconds.
    
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    return capacitance * resistance

@returns_unit("h")
def capacitor_lifetime(temp, nominal_lifetime="2000 h", nominal_lifetime_temperature="105 °C", A=10.):
    """
    Estimate the lifetime of a capacitor given its working temperature,
    
    its nominal lifetime at a nominal lifetime temperature, and coefficient A.

    Coefficient A is the temperature difference for which to assume a halving of the lifetime.

    Based on:
    https://www.illinoiscapacitor.com/tech-center/life-calculators.aspx
    """
    temp = normalize_temperature(temp)
    nominal_lifetime_temperature = normalize_temperature(nominal_lifetime_temperature)
    nominal_lifetime = normalize_numeric(nominal_lifetime)
    # Compute lifetime
    tdelta = temp - nominal_lifetime_temperature
    return nominal_lifetime * 2**(-(tdelta/A))

@returns_unit("J")
def capacitor_energy(capacitance: CapacitanceFarad, voltage: VoltageV):
    """
    Compute the total energy stored in a capacitor given:
    
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The energy is returned as joules.
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    return 0.5 * capacitance * np.square(voltage)

@returns_unit("C")
def capacitor_charge(capacitance: CapacitanceFarad, voltage: VoltageV):
    """
    Compute the total charge stored in a capacitor given:
    
    - The capacitance in farads
    - The voltage the capacitor is charged to
    The charge is returned in coulombs.
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    return capacitance * voltage

@returns_unit("V")
def capacitor_voltage_by_energy(capacitance: CapacitanceFarad, energy: EnergyJ, starting_voltage="0V"):
    """
    Compute the voltage of a capacitor given:
    
    - The capacitance in farads
    - The energy stored in joules
    The voltage is returned in volts.
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    energy = normalize_energy(energy) if isinstance(energy, str) else energy
    # Compute starting energy
    starting_energy = capacitor_energy(capacitance, starting_voltage)
    # Compute voltage
    return np.sqrt(2 * (energy + starting_energy) / capacitance)

@returns_unit("s")
def capacitor_constant_current_discharge_time(capacitance: CapacitanceFarad, initial_voltage: VoltageV, current: CurrentA, target_voltage="0V"):
    """
    Compute the time it takes to charge a capacitor to [target_voltage]
    
    using a constant current.

    Keyword Arguments
    -----------------
    capacitance : number or Engineer string
        The capacitance of the capacitor in farads.
    voltage : number or Engineer string
        The initial voltage of the capacitor in volts.
    current : number or Engineer string
        The charge current in amperes.
    target_voltage : number or Engineer string, optional
        The target voltage to discharge the capacitor to.

    Returns
    -------
    float
        The time in seconds.
    
    """
    # Use charge function with "negative current"
    # Since from the view of the charge function, its generating a negative
    # voltage charge, this will result in a positive time
    return capacitor_constant_current_charge_time(capacitance, target_voltage, current, initial_voltage)

@returns_unit("s")
def capacitor_constant_current_charge_time(capacitance: CapacitanceFarad, target_voltage: VoltageV, current: CurrentA, initial_voltage="0V"):
    """
    Compute the time it takes to charge a capacitor to [target_voltage]
    
    using a constant current.

    Keyword Arguments
    -----------------
    capacitance : number or Engineer string
        The capacitance of the capacitor in farads.
    initial_voltage : number or Engineer string
        The initial voltage of the capacitor in volts.
    current : number or Engineer string
        The discharge current in amperes.
    target_voltage : number or Engineer string, optional
        The target voltage to discharge the capacitor to.

    Returns
    -------
    float
        The time in seconds.
    
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    target_voltage = normalize_voltage(target_voltage) if isinstance(target_voltage, str) else target_voltage
    current = normalize_current(current) if isinstance(current, str) else current
    initial_voltage = normalize_voltage(initial_voltage) if isinstance(initial_voltage, str) else initial_voltage
    return capacitance * (initial_voltage - target_voltage) / current


@returns_unit("s")
def capacitor_resistor_charge_time(capacitance: CapacitanceFarad, resistance: ResistanceOhm, source_voltage: VoltageV, target_voltage: VoltageV, initial_voltage="0V", diode_model=None, diode_voltage=None):
    """
    Compute the time it takes to charge a capacitor through a resistor.

    Parameters:
    - capacitance: The capacitance in farads.
    - resistance: The charging resistance in ohms.
    - source_voltage: The source voltage in volts.
    - target_voltage: The target capacitor voltage in volts.
    - initial_voltage: The initial capacitor voltage in volts.
    - diode_model: Optional diode model, or a scalar/string forward voltage for SimpleDiodeModel.
    - diode_voltage: Backward-compatible alias for diode_model.

    Returns:
    The time in seconds.

    The capacitor asymptotically approaches source_voltage - diode_model.minimum_series_voltage().
    
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    source_voltage = normalize_voltage(source_voltage) if isinstance(source_voltage, str) else source_voltage
    target_voltage = normalize_voltage(target_voltage) if isinstance(target_voltage, str) else target_voltage
    initial_voltage = normalize_voltage(initial_voltage) if isinstance(initial_voltage, str) else initial_voltage
    diode_model = normalize_diode_model(diode_voltage if diode_voltage is not None else diode_model)
    minimum_series_voltage = normalize_numeric(diode_model.minimum_series_voltage())
    final_voltage = source_voltage - minimum_series_voltage
    if np.any(final_voltage < initial_voltage):
        raise ValueError("source_voltage - diode minimum voltage must be greater than or equal to initial_voltage")
    if np.any(target_voltage < initial_voltage):
        raise ValueError("target_voltage must be greater than or equal to initial_voltage")
    if np.any(target_voltage > final_voltage):
        raise ValueError("target_voltage must be less than or equal to source_voltage - diode minimum voltage")
    return _capacitor_resistor_model_time(
        capacitance,
        resistance,
        source_voltage - initial_voltage,
        source_voltage - target_voltage,
        diode_model,
        initial_voltage,
        target_voltage,
    )


@returns_unit("s")
def capacitor_resistor_discharge_time(capacitance: CapacitanceFarad, resistance: ResistanceOhm, initial_voltage: VoltageV, target_voltage="0V", diode_model=None, diode_voltage=None):
    """
    Compute the time it takes to discharge a capacitor through a resistor.

    Parameters:
    - capacitance: The capacitance in farads.
    - resistance: The discharge resistance in ohms.
    - initial_voltage: The initial capacitor voltage in volts.
    - target_voltage: The target capacitor voltage in volts.
    - diode_model: Optional diode model, or a scalar/string forward voltage for SimpleDiodeModel.
    - diode_voltage: Backward-compatible alias for diode_model.

    Returns:
    The time in seconds.

    The capacitor asymptotically approaches diode_model.minimum_series_voltage().
    
    """
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    initial_voltage = normalize_voltage(initial_voltage) if isinstance(initial_voltage, str) else initial_voltage
    target_voltage = normalize_voltage(target_voltage) if isinstance(target_voltage, str) else target_voltage
    diode_model = normalize_diode_model(diode_voltage if diode_voltage is not None else diode_model)
    final_voltage = normalize_numeric(diode_model.minimum_series_voltage())
    if np.any(initial_voltage < final_voltage):
        raise ValueError("initial_voltage must be greater than or equal to diode minimum voltage")
    if np.any(target_voltage > initial_voltage):
        raise ValueError("target_voltage must be less than or equal to initial_voltage")
    if np.any(target_voltage < final_voltage):
        raise ValueError("target_voltage must be greater than or equal to diode minimum voltage")
    return _capacitor_resistor_model_time(
        capacitance,
        resistance,
        initial_voltage,
        target_voltage,
        diode_model,
        initial_voltage,
        target_voltage,
    )

@returns_unit("F")
def parallel_plate_capacitors_capacitance(area, distance, epsilon: PermittivityFm):
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
    epsilon = normalize_permittivity(epsilon) if isinstance(epsilon, str) else epsilon
    return epsilon * area / distance

@returns_unit("F")
def capacitor_capacitance_by_energy(energy: EnergyJ, voltage: VoltageV, starting_voltage="0V"):
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
    energy = normalize_energy(energy) if isinstance(energy, str) else energy
    voltage = normalize_voltage(voltage) if isinstance(voltage, str) else voltage
    starting_voltage = normalize_voltage(starting_voltage) if isinstance(starting_voltage, str) else starting_voltage
    voltage_squared_diff = np.square(voltage) - np.square(starting_voltage)
    return 2 * energy / voltage_squared_diff

@returns_unit("J")
def capacitor_charging_energy(capacitance: CapacitanceFarad, end_voltage: VoltageV, starting_voltage="0V"):
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
