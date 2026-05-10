#!/usr/bin/env python3
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics.Frequency import FrequencyHz, normalize_frequency
from ._normalize import normalize_with_known_units
import scipy.constants

__all__ = [
    "NucleusLarmorFrequency",
    "larmor_frequency",
    "normalize_magnetic_field", "MagneticFieldTesla",
    "normalize_frequency", "FrequencyHz"
]

# Nucleus Larmor frequencies in Hz

class NucleusLarmorFrequency:
    
    """Standard frequencies for common nuclei."""
    
    H1 = scipy.constants.physical_constants['shielded proton gyromag. ratio in MHz/T'][0]
    He3 = scipy.constants.physical_constants['shielded helion gyromag. ratio in MHz/T'][0]

def normalize_magnetic_field(field: NormalizableArgument) -> NormalizedComputable:
    return normalize_with_known_units(field, {"T": 1.0, "Tesla": 1.0, "tesla": 1.0, "mT": 1e-3, "µT": 1e-6, "G": 1e-4, "Gauss": 1e-4, "gauss": 1e-4}, quantity_name="magnetic field")

MagneticFieldTesla = Annotated[NormalizedComputable, normalize_magnetic_field]

@returns_unit("Hz")
def larmor_frequency(b0: MagneticFieldTesla, nucleus_larmor_frequency=NucleusLarmorFrequency.H1):
    """Get the magnetic resonance frequency (larmor frequency).
    
    for a given nucleus in a given magnetic field strength B0.

    Note that the frequency is given in Hz, not in MHz!

    Parameters
    ----------
    b0 : MagneticFieldTesla
        Magnetic field strength in Tesla.
    nucleus_larmor_frequency : float, optional
        Larmor frequency of the nucleus in MHz/T.

    Returns
    -------
    float
        Larmor frequency in Hz.

    """
    b0 = normalize_magnetic_field(b0)
    return b0 * (nucleus_larmor_frequency * 1e6) # MHz/T -> Hz/T