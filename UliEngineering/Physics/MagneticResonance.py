#!/usr/bin/env python3
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
import scipy.constants

__all__ = [
    "NucleusLarmorFrequency",
    "larmor_frequency"
]

# Nucleus Larmor frequencies in Hz

class NucleusLarmorFrequency:
    """Standard frequencies for common nuclei"""
    H1 = scipy.constants.physical_constants['shielded proton gyromag. ratio in MHz/T'][0]
    He3 = scipy.constants.physical_constants['shielded helion gyromag. ratio in MHz/T'][0]

@returns_unit("Hz")
@normalize_numeric_args
def larmor_frequency(b0, nucleus_larmor_frequency=NucleusLarmorFrequency.H1):
    """
    Get the magnetic resonance frequency (larmor frequency)
    for a given nucleus in a given magnetic field strength B0
    
    Note that the frequency is given in Hz, not in MHz!
    
    :param b0: Magnetic field strength in Tesla
    :param nucleus_larmor_frequency: Larmor frequency of the nucleus in MHz/T
    """
    return b0 * (nucleus_larmor_frequency * 1e6) # MHz/T -> Hz/T