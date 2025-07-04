#!/usr/bin/env python3
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args
import numpy as np

__all__ = [
    "logarithmic_amplifier_output_voltage",
    "logarithmic_amplifier_input_current"
]

@normalize_numeric_args
def logarithmic_amplifier_output_voltage(ipd, gain, intercept):
    """
    Compute the logarithmic output voltage of a ADL5303 (probably its more general than that)
    
    According to Formula (1) in the [ADL5303 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adl5303.pdf)
    
    Parameters
    ----------
    ipd : float
        The input current (in Amperes)
    gain : float
        The gain (volts per decade) 
    intercept : float
        The intercept point (in Amperes)
        
    Returns: The output voltage in Volts
    """
    return gain * np.log10(ipd / intercept)

@normalize_numeric_args
def logarithmic_amplifier_input_current(vout, gain, intercept):
    """
    Compute the input current based on the output voltage of a logarithmic amplifier
    
    The formula for this is Ipd = intercept * 10^(vout / Gain)
    https://techoverflow.net/2024/09/23/how-to-compute-the-input-current-of-a-logarithmic-amplifier/
    
    Parameters
    ----------
    vout : float
        The output voltage of the logarithmic amplifier
    gain : float
        The gain (volts per decade)
    intercept : float
        The intercept point (in Amperes)
    """
    return intercept * np.power(10, vout / gain)
