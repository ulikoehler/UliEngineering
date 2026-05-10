#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermistor computations

For reference see e.g.
https://www.electronics-tutorials.ws/io/thermistors.html
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Types import NormalizableArgument
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.Physics.Temperature import normalize_temperature_kelvin
from .Diode import normalize_resistance, ResistanceOhm
import numpy as np
from UliEngineering.Physics.Temperature import kelvin_to_celsius


__all__ = [
    "thermistor_b_value",
    "thermistor_temperature",
    "thermistor_resistance",
]

def thermistor_b_value(r1: ResistanceOhm, r2: ResistanceOhm, t1: NormalizableArgument = 25.0, t2: NormalizableArgument = 100.0):
    """
    Compute the B value of a thermistor given its resistance at two temperatures

    The formula is B = (T1*T2) / (T2-T1) * ln(R1/R2)
    with T1 and T2 being the temperatures in Kelvin and R1 and R2 being the resistances

    t1/t2 can be given either as strings e.g. "0°F", "100°C", "300K" or as numbers
    r1/r2 can be given either as strings e.g. "1kΩ", "1MΩ" or as numbers

    Returns the B value (unitless)
    """
    # Normalize to Kelvin (temperature needs special handling)
    t1 = normalize_temperature_kelvin(t1)
    t2 = normalize_temperature_kelvin(t2)
    r1 = normalize_resistance(r1) if isinstance(r1, str) else r1
    r2 = normalize_resistance(r2) if isinstance(r2, str) else r2

    return (t1*t2) / (t2-t1) * np.log(r1/r2)

@returns_unit("°C")
def thermistor_temperature(resistance: ResistanceOhm, beta: NormalizableArgument = 3950.0, R0: ResistanceOhm = 100e3, T0: NormalizableArgument = 25.0):
    """
    Calculate the temperature of a NTC thermistor using the Beta parameter model.

    Parameters:
    - resistance: The measured resistance of the thermistor in Ohms, for which to calculate the temperature.
    - beta: The Beta constant of the thermistor.
    - R0: The resistance of the thermistor at reference temperature T0 (default is 10kOhms).
    - T0: The reference temperature in Celsius (default is 25°C).

    Returns:
    - Temperature in degrees.
    """
    R0 = normalize_resistance(R0) if isinstance(R0, str) else R0
    T0 = normalize_temperature_kelvin(T0)
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    beta = normalize_numeric(beta) if isinstance(beta, str) else beta
    temperature_kelvin = 1 / (1/T0 + (1/beta) * np.log(resistance/R0))
    return kelvin_to_celsius(temperature_kelvin)

@returns_unit("Ω")
def thermistor_resistance(temperature: NormalizableArgument, beta: NormalizableArgument = 3950.0, R0: ResistanceOhm = 100e3, T0: NormalizableArgument = 25.0):
    """
    Calculate the resistance of a thermistor given its temperature.

    Parameters:
    temperature (float): The temperature in Kelvin
    A, B, C (float): The Steinhart-Hart coefficients for the thermistor

    Returns:
    float: The resistance of the thermistor in Ohms
    """
    temperature_kelvin = normalize_temperature_kelvin(temperature)
    t0_kelvin = normalize_temperature_kelvin(T0)
    R0 = normalize_resistance(R0) if isinstance(R0, str) else R0
    beta = normalize_numeric(beta) if isinstance(beta, str) else beta
    # Calculate the resistance using the inverse Steinhart-Hart equation
    # Wolfram Alpha: solve K = 1 / (1/T + (1/b) * log(R/R0)) for R
    resistance = R0 * np.exp(beta * (1/temperature_kelvin - 1/t0_kelvin))
    return resistance