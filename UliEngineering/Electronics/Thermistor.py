#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermistor computations

For reference see e.g.
https://www.electronics-tutorials.ws/io/thermistors.html
"""
from UliEngineering.EngineerIO import normalize_engineer_notation, normalize_numeric
from UliEngineering.Units import Unit
from UliEngineering.Physics.Temperature import normalize_temperature_kelvin
from UliEngineering.Exceptions import InvalidUnitException
import numpy as np
from UliEngineering.Physics.Temperature import zero_Celsius, kelvin_to_celsius


__all__ = [
    "thermistor_b_value",
    "thermistor_temperature",
    "thermistor_resistance",
]

def thermistor_b_value(r1, r2, t1=25.0, t2=100.0):
    """
    Compute the B value of a thermistor given its resistance at two temperatures
    
    The formula is B = (T1*T2) / (T2-T1) * ln(R1/R2)
    with T1 and T2 being the temperatures in Kelvin and R1 and R2 being the resistances
    
    t1/t2 can be given either as strings e.g. "0°F", "100°C", "300K" or as numbers
    r1/r2 can be given either as strings e.g. "1kΩ", "1MΩ" or as numbers
    
    Returns the B value (unitless)
    """
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    # Normalize to Kelvin
    t1 = normalize_temperature_kelvin(t1)
    t2 = normalize_temperature_kelvin(t2)
    print(t1, t2, r1, r2)
   
    return (t1*t2) / (t2-t1) * np.log(r1/r2)

def thermistor_temperature(resistance, beta=3950.0, R0=100e3, T0=25.0) -> Unit("°C"):
    """
    Calculate the temperature of a NTC thermistor using the Beta parameter model.
    
    Parameters:
    - resistance: The measured resistance of the thermistor in Ohms, for which to calculate the temperature.
    - beta: The Beta constant of the thermistor.
    - c: An additional constant, currently unused.
    - R0: The resistance of the thermistor at reference temperature T0 (default is 10kOhms).
    - T0: The reference temperature in Celsius (default is 25°C).
    
    Returns:
    - Temperature in degrees.
    """
    resistance = normalize_numeric(resistance)
    R0 = normalize_numeric(R0)
    T0 = normalize_temperature_kelvin(T0)
    temperature_kelvin = 1 / (1/T0 + (1/beta) * np.log(resistance/R0))
    return kelvin_to_celsius(temperature_kelvin)

def thermistor_resistance(temperature, beta=3950.0, R0=100e3, T0=25.0) -> Unit("Ω"):
    """
    Calculate the resistance of a thermistor given its temperature.

    Parameters:
    temperature (float): The temperature in Kelvin
    A, B, C (float): The Steinhart-Hart coefficients for the thermistor

    Returns:
    float: The resistance of the thermistor
    """
    temperature_kelvin = normalize_temperature_kelvin(temperature)
    t0_kelvin = normalize_temperature_kelvin(T0)
    # Calculate the resistance using the inverse Steinhart-Hart equation
    # Wolfram Alpha: solve K = 1 / (1/T + (1/b) * log(R/R0)) for R
    resistance = R0 * np.exp(beta * (1/temperature_kelvin - 1/t0_kelvin))
    return resistance