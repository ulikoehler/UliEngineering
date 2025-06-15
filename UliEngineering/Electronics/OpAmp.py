#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for operational amplifier calculations.

Usage example:
>>> from UliEngineering.Electronics.OpAmp import summing_amplifier_noninv
>>> # Example: sum 2.5V and 0.5V with a total sum-referred gain of 1.0
>>> formatValue(summing_amplifier_noninv(
        "2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ"), "V"))

"""

__all__ = [
    "summing_amplifier_noninv",
    "noninverting_amplifier_gain"
]

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit


@returns_unit("V")
@normalize_numeric_args
def summing_amplifier_noninv(v1, v2, r1, r2, rfb1, rfb2):
    """
    Computes the output voltage of a non-inverting summing amplifier:
    V1 connected via R1 to IN+
    V2 connected via R2 to IN+
    IN- connected via RFB1 to GND
    IN- connected via RFB2 to VOut
    """
    return (1.0 + rfb2 / rfb1) * (v1 * (r2 / (r1 + r2)) + v2 * (r1 / (r1 + r2)))

@returns_unit("V/V")
@normalize_numeric_args
def noninverting_amplifier_gain(r1, r2):
    """
    Computes the gain of a non-inverting amplifier with feedback resistors R1 and R2.
    
    # 2D ASCII graphic with rectangular opamp
    
    R1 is the resistor connected between the OpAmp output and the OpAmp IN(-).
    R2 is the resistor connected between the OpAmp IN(-) and GND.
    
    R2 can also be infinity (np.inf), in which case the gain is 1.0.
    """
    return 1.0 + r1 / r2
