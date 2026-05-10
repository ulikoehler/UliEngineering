#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for operational amplifier calculations.

Usage example:
>>> from UliEngineering.Electronics.OpAmp import summing_amplifier_noninv
>>> # Example: sum 2.5V and 0.5V with a total sum-referred gain of 1.0
>>> formatValue(summing_amplifier_noninv(
        "2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ"), "V"))

"""

from UliEngineering.EngineerIO.Decorators import returns_unit
from .Diode import normalize_voltage, VoltageV, normalize_resistance, ResistanceOhm

__all__ = [
    "summing_amplifier_noninv",
    "noninverting_amplifier_gain"
]


@returns_unit("V")
def summing_amplifier_noninv(v1: VoltageV, v2: VoltageV, r1: ResistanceOhm, r2: ResistanceOhm, rfb1: ResistanceOhm, rfb2: ResistanceOhm):
    """
    Computes the output voltage of a non-inverting summing amplifier.

    V1 connected via R1 to IN+.
    V2 connected via R2 to IN+.
    IN- connected via RFB1 to GND.
    IN- connected via RFB2 to VOut.

    Parameters
    ----------
    v1 : VoltageV
        First input voltage.
    v2 : VoltageV
        Second input voltage.
    r1 : ResistanceOhm
        Resistor for V1 input.
    r2 : ResistanceOhm
        Resistor for V2 input.
    rfb1 : ResistanceOhm
        Feedback resistor to ground.
    rfb2 : ResistanceOhm
        Feedback resistor to output.

    Returns
    -------
    float
        Output voltage in Volts.
    
    """
    v1 = normalize_voltage(v1) if isinstance(v1, str) else v1
    v2 = normalize_voltage(v2) if isinstance(v2, str) else v2
    r1 = normalize_resistance(r1) if isinstance(r1, str) else r1
    r2 = normalize_resistance(r2) if isinstance(r2, str) else r2
    rfb1 = normalize_resistance(rfb1) if isinstance(rfb1, str) else rfb1
    rfb2 = normalize_resistance(rfb2) if isinstance(rfb2, str) else rfb2
    return (1.0 + rfb2 / rfb1) * (v1 * (r2 / (r1 + r2)) + v2 * (r1 / (r1 + r2)))

@returns_unit("V/V")
def noninverting_amplifier_gain(r1: ResistanceOhm, r2: ResistanceOhm):
    """
    Computes the gain of a non-inverting amplifier with feedback resistors R1 and R2.

    R1 is the resistor connected between the OpAmp output and the OpAmp IN(-).
    R2 is the resistor connected between the OpAmp IN(-) and GND.

    R2 can also be infinity (np.inf), in which case the gain is 1.0.

    Parameters
    ----------
    r1 : ResistanceOhm
        Feedback resistor from output to IN(-).
    r2 : ResistanceOhm
        Resistor from IN(-) to ground.

    Returns
    -------
    float
        Amplifier gain (V/V).

    """
    r1 = normalize_resistance(r1) if isinstance(r1, str) else r1
    r2 = normalize_resistance(r2) if isinstance(r2, str) else r2
    return 1.0 + r1 / r2
