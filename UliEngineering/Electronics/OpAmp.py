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
import itertools
import math
from UliEngineering.EngineerIO import autoNormalizeEngineerInput

def summing_amplifier_noninv(v1, v2, r1, r2, rfb1, rfb2):
    """
    Computes the output voltage of a non-inverting summing amplifier:
    V1 connected via R1 to IN+
    V2 connected via R2 to IN+
    IN- connected via RFB1 to GND
    IN- connected via RFB2 to VOut
    """
    v1, _ = autoNormalizeEngineerInput(v1)
    v2, _ = autoNormalizeEngineerInput(v2)
    r1, _ = autoNormalizeEngineerInput(r1)
    r2, _ = autoNormalizeEngineerInput(r2)
    rfb1, _ = autoNormalizeEngineerInput(rfb1)
    rfb2, _ = autoNormalizeEngineerInput(rfb2)
    return (1.0 + rfb2 / rfb1) * (v1 * (r2 / (r1 + r2)) + v2 * (r1 / (r1 + r2)))
