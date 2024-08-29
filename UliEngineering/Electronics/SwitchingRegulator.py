#!/usr/bin/env python3
"""
Utilities for computing switching regulator parameters
"""
from UliEngineering.EngineerIO import normalize_numeric, Unit

def buck_regulator_inductance(vin, vout, frequency, ioutmax, K=0.3) -> Unit("H"):
    """
    ## Inductor selection algorithm
    
    Compute the optimal inducitivity of a buck regulator
    
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