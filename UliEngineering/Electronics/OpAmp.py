#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for operational amplifier calculations.

Usage example:
>>> from UliEngineering.Electronics.OpAmp import summing_amplifier_noninv
>>> # Example: sum 2.5V and 0.5V with a total sum-referred gain of 1.0
>>> formatValue(summing_amplifier_noninv(
        "2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ"), "V"))

"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

def summing_amplifier_noninv(v1, v2, r1, r2, rfb1, rfb2) -> Unit("V"):
    """
    Computes the output voltage of a non-inverting summing amplifier:
    V1 connected via R1 to IN+
    V2 connected via R2 to IN+
    IN- connected via RFB1 to GND
    IN- connected via RFB2 to VOut
    """
    v1 = normalize_numeric(v1)
    v2 = normalize_numeric(v2)
    r1 = normalize_numeric(r1)
    r2 = normalize_numeric(r2)
    rfb1 = normalize_numeric(rfb1)
    rfb2 = normalize_numeric(rfb2)
    return (1.0 + rfb2 / rfb1) * (v1 * (r2 / (r1 + r2)) + v2 * (r1 / (r1 + r2)))
