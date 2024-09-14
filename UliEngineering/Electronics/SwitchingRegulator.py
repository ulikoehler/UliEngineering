#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing switching regulator parameters
"""
from UliEngineering.EngineerIO import normalize_numeric, Unit
from collections import namedtuple

__all__ = [
    "buck_regulator_inductance", "buck_regulator_inductor_current", "InductorCurrent",
    "buck_regulator_duty_cycle", "buck_regulator_inductor_ripple_current",
    "buck_regulator_inductor_peak_current", "buck_regulator_inductor_rms_current"]

def buck_regulator_inductance(vin, vout, frequency, ioutmax, K=0.3) -> Unit("H"):
    """
    Compute the optimal inductance for use in a buck regulator
    
    This formula is based on the the inductor ripple current fraction [K].
    
    The formula we use is:
    
    L = ((vin - vout) * (vout) / (f * K * Ioutmax)) * (Vout/Vin)
    
    (note that Vout/Vin is an estimation for the duty cycle.)
    
    A good assumption which is shared by most major manufacturers is
    to choose the inductor value in between K=0.2 and K=0.4.
    Typically, the best inductor value is around K=0.3,
    but one depends 
    
    It is generally recommended by the more verbose datasheets, to alwas choose
    the inductor larger than the value obtained with K=0.1. This is due to the
    current mode control scheme which requires a certain level of inductor ripple.
    
    Note that many datasheets also specify minimum inductor values to avoid
    subharmonic oscillations. This depends on the part and varies by more than
    and order of magnitude and is not handled by the function.
    
    For reference see e.g. TI at https://www.ti.com/lit/ds/symlink/lmr36006.pdf#page=22,
    section 9.2.1.2.4: Inductor Selection.
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    frequency = normalize_numeric(frequency)
    ioutmax = normalize_numeric(ioutmax)
    K = normalize_numeric(K)
    return ((vin - vout) * (vout) / (frequency * K * ioutmax)) * (vout/vin)

InductorCurrent = namedtuple("InductorCurrent", ["peak", "rms"])

def buck_regulator_duty_cycle(vin, vout) -> float:
    """
    Estimate the duty cycle of a buck regulator

    D = Vout/Vin
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    return vout / vin

def buck_regulator_inductor_ripple_current(vin, vout, inductance, frequency, ioutmax) -> Unit("A"):
    """
    Compute the ripple current ΔIL in the inductor
    
    This can be used to determine the peak current rating of the inductor.
    
    The formula is:
    
    ΔIL = (Vin - Vout) * D / (L * frequency)
    where D = Vout/Vin
    
    Returns the ripple current in Amperes
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    inductance = normalize_numeric(inductance)
    frequency = normalize_numeric(frequency)
    ioutmax = normalize_numeric(ioutmax)
    D = buck_regulator_duty_cycle(vin, vout)
    return (vin - vout) * D / (inductance * frequency)

def buck_regulator_inductor_current(vin, vout, inductance, frequency, ioutmax) -> InductorCurrent:
    """
    Compute an estimation for the peak inductor current.
    
    This can be used to determine inductor value

    This approach is based on the formula found in the LM76002 datasheet
    from Texas instruments:
    https://www.ti.com/lit/ds/symlink/lm76002.pdf
    
    D = (Vout/Vin) # Duty cycle estimation
    ΔIL = (Vin - Vout) * D / (L * frequency)
    
    Ilpeak = Ioutmax + ΔIL / 2
    Ilrms = sqrt(Ioutmax^2 + ΔIL^2 / 12)
    
    Returns an InductorCurrent namedtuple with the peak and RMS current (unit: Amperes)
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    inductance = normalize_numeric(inductance)
    frequency = normalize_numeric(frequency)
    ioutmax = normalize_numeric(ioutmax)
    D = buck_regulator_duty_cycle(vin, vout)
    ΔIL = buck_regulator_inductor_ripple_current(vin, vout, inductance, frequency, ioutmax)
    Ilpeak = ioutmax + ΔIL / 2
    Ilrms = (ioutmax**2 + ΔIL**2 / 12)**0.5
    return InductorCurrent(peak=Ilpeak, rms=Ilrms)

def buck_regulator_inductor_peak_current(vin, vout, inductance, frequency, ioutmax, safety_factor=1.2) -> Unit("A"):
    """
    Compute the peak inductor current rating
    
    This can be used to determine the saturation current rating of the inductor.
    Especially ferrite core inductors should have sufficient saturation current rating
    to accomodate the maximum peak current for the worst-case operating condition.
    
    The formula is:
    
    Ilpeak = Ioutmax + ΔIL / 2
    where ΔIL = (Vin - Vout) * D / (L * frequency)
    and D = Vout/Vin
    
    Returns the peak inductor current rating in Ampere,
    including the safety factor.
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    inductance = normalize_numeric(inductance)
    frequency = normalize_numeric(frequency)
    ioutmax = normalize_numeric(ioutmax)
    safety_factor = normalize_numeric(safety_factor)
    D = buck_regulator_duty_cycle(vin, vout)
    ΔIL = buck_regulator_inductor_ripple_current(vin, vout, inductance, frequency, ioutmax)
    Ilpeak = ioutmax + ΔIL / 2
    return Ilpeak * safety_factor

def buck_regulator_inductor_rms_current(vin, vout, inductance, frequency, ioutmax, safety_factor=1.2) -> Unit("A"):
    """
    Compute the RMS inductor current rating
    
    This can be used to determine the RMS current rating of the inductor.
    The required RMS current rating is typically lower than the peak current rating,
    and this fact can be used to select a smaller-sized inductor.
    
    The formula is:
    
    Ilrms = sqrt(Ioutmax^2 + ΔIL^2 / 12)
    where ΔIL = (Vin - Vout) * D / (L * frequency)
    and D = Vout/Vin
    
    Returns the RMS inductor current rating in Ampere,
    including the safety factor.
    """
    vin = normalize_numeric(vin)
    vout = normalize_numeric(vout)
    inductance = normalize_numeric(inductance)
    frequency = normalize_numeric(frequency)
    ioutmax = normalize_numeric(ioutmax)
    safety_factor = normalize_numeric(safety_factor)
    D = buck_regulator_duty_cycle(vin, vout)
    ΔIL = buck_regulator_inductor_ripple_current(vin, vout, inductance, frequency, ioutmax)
    Ilrms = (ioutmax**2 + ΔIL**2 / 12)**0.5
    return Ilrms * safety_factor