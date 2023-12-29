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
    
    