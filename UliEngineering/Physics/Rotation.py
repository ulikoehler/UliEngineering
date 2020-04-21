#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for acceleration
"""
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit
import numpy as np
import scipy.constants

__all__ = ["rpm_to_Hz", "rpm_to_rps", "hz_to_rpm"]

def rpm_to_Hz(rpm) -> Unit("Hz"):
    """
    Compute the rotational speed in Hz given the rotational speed in rpm
    """
    rpm = normalize_numeric(rpm)
    return rpm / 60.

def hz_to_rpm(hz) -> Unit("rpm"):
    """
    Compute the rotational speed in rpm given the rotational speed in Hz
    """
    hz = normalize_numeric(hz)
    return hz * 60.

rpm_to_rps = rpm_to_Hz
