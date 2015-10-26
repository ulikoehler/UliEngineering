#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for operational amplifier calculations
"""
import itertools
import math
from .EngineerIO import *

def summing_amplifier_noninv(v1, v2, r1, r2, rfb1, rfb2):
    """
    Computes the output voltage of a non-inverting summing amplifier:
    V1 connected via R1 to IN+
    V2 connected via R2 to IN+
    IN- connected via RFB1 to GND
    IN- connected via RFB2 to VOut
    """
    v1, _ = normalizeEngineerInputIfStr(v1)
    v2, _ = normalizeEngineerInputIfStr(v2)
    r1, _ = normalizeEngineerInputIfStr(r1)
    r2, _ = normalizeEngineerInputIfStr(r2)
    rfb1, _ = normalizeEngineerInputIfStr(rfb1)
    rfb2, _ = normalizeEngineerInputIfStr(rfb2)
    return (1.0 + rfb2 / rfb1) * (v1 * (r2 / (r1 + r2)) + v2 * (r1 / (r1 + r2)))

if __name__ == "__main__":
    # Example: sum 2.5V and 0.5V with a total sum-referred gain of 1.0
    print(formatValue(summing_amplifier_noninv(
        "2.5V", "500mV", "1kΩ", "1kΩ", "1kΩ", "1kΩ"), "V"))
