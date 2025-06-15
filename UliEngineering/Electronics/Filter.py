#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electronic filter and time constant utilities
"""
from UliEngineering.EngineerIO import normalize_numeric
from collections import namedtuple
import numpy as np

from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit

__all__ = [
    "lc_cutoff_frequency", "rc_cutoff_frequency",
    "rc_feedforward_pole_and_zero", "PoleAndZero",
    "rc_time_constant", "rl_time_constant", "rl_cutoff_frequency",
    "rc_charge_time", "rc_discharge_time", "rl_current_rise_time", 
    "rl_current_fall_time", "rc_step_response", "rl_step_response", 
    "rlc_resonant_frequency", "rlc_quality_factor", "rlc_damping_ratio", 
    "rlc_bandwidth"
]

@returns_unit("Hz")
@normalize_numeric_args
def lc_cutoff_frequency(l, c):
    """
    Compute the resonance frequency of an LC oscillator circuit
    given the inductance and capacitance.
    
    This function can likewise be used to compute the corner frequency f_p
    of a LC filter.

    Parameters
    ----------
    l : float
        The inductance in Henry
    c : float
        The capacitance in Farad
    """
    return 1. / (2 * np.pi * np.sqrt(l * c))

@returns_unit("Hz")
@normalize_numeric_args
def rc_cutoff_frequency(r, c):
    """
    Compute the corner frequency of an RC filter given the resistance
    and capacitance.

    Parameters
    ----------
    r : float
        The resistance in Ohm
    c : float
        The capacitance in Farad
    """
    return 1. / (2 * np.pi * r * c)

PoleAndZero = namedtuple("PoleAndZero", ["pole", "zero"])

@normalize_numeric_args
def rc_feedforward_pole_and_zero(r1, r2, cff):
    """
    Compute the pole and zero of a resistor divider with a feedforward capacitor.
    This is useful to compute the compensation capacitor.
    
    Cff is assumed to be in parallel with R1, while R2 goes to ground.
    
    For reference, see
    https://www.ti.com/lit/an/slva289b/slva289b.pdf
    equations 1 and 2.

    Parameters
    ----------
    r1 : float
        The resistance of the feedback path in Ohm
    r2 : float
        The resistance of the feedforward path in Ohm
    cff : float
        The capacitance of the feedforward path in Farad
    """
    return PoleAndZero(
        zero=rc_cutoff_frequency(r1, cff),
        pole=(1./r1 + 1./r2)/(2*np.pi*cff)
    )

@returns_unit("s")
@normalize_numeric_args
def rc_time_constant(resistance, capacitance):
    """
    Calculate the time constant (τ) of an RC circuit.
    
    τ = R × C
    
    The time constant represents the time required for the voltage
    across the capacitor to reach approximately 63.2% of its final value
    when charging, or to decay to 36.8% when discharging.
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    capacitance : number or Engineer string
        The capacitance value in Farads
        
    Returns
    -------
    float
        Time constant in seconds
    """
    return resistance * capacitance

@returns_unit("s")
@normalize_numeric_args
def rl_time_constant(resistance, inductance):
    """
    Calculate the time constant (τ) of an RL circuit.
    
    τ = L / R
    
    The time constant represents the time required for the current
    through the inductor to reach approximately 63.2% of its final value
    when energizing, or to decay to 36.8% when de-energizing.
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
        
    Returns
    -------
    float
        Time constant in seconds
    """
    return inductance / resistance

@returns_unit("Hz")
@normalize_numeric_args
def rl_cutoff_frequency(resistance, inductance):
    """
    Calculate the cutoff frequency (fc) of an RL circuit.
    
    fc = R / (2π × L)
    
    This is the -3dB frequency where the output is 70.7% of the input
    in a low-pass or high-pass RL filter.
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
        
    Returns
    -------
    float
        Cutoff frequency in Hz
    """
    return resistance / (2 * np.pi * inductance)

@returns_unit("s")
@normalize_numeric_args
def rc_charge_time(resistance, capacitance, initial_voltage, final_voltage, target_voltage):
    """
    Calculate the time required for a capacitor to charge from initial_voltage
    to target_voltage when charging towards final_voltage through a resistor.
    
    t = -R × C × ln((target_voltage - final_voltage) / (initial_voltage - final_voltage))
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    capacitance : number or Engineer string
        The capacitance value in Farads
    initial_voltage : number or Engineer string
        The initial voltage across the capacitor in Volts
    final_voltage : number or Engineer string
        The final (asymptotic) voltage in Volts
    target_voltage : number or Engineer string
        The target voltage to reach in Volts
        
    Returns
    -------
    float
        Time in seconds
    """
    initial_voltage = normalize_numeric(initial_voltage)
    final_voltage = normalize_numeric(final_voltage)
    target_voltage = normalize_numeric(target_voltage)
    
    if abs(initial_voltage - final_voltage) < 1e-12:
        return 0.0  # Already at final voltage
    if abs(target_voltage - final_voltage) < 1e-12:
        return float('inf')  # Never reaches final voltage
    
    return -resistance * capacitance * np.log((target_voltage - final_voltage) / (initial_voltage - final_voltage))

@returns_unit("s")
@normalize_numeric_args
def rc_discharge_time(resistance, capacitance, initial_voltage, target_voltage):
    """
    Calculate the time required for a capacitor to discharge from initial_voltage
    to target_voltage through a resistor (assuming discharge to 0V).
    
    t = -R × C × ln(target_voltage / initial_voltage)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    capacitance : number or Engineer string
        The capacitance value in Farads
    initial_voltage : number or Engineer string
        The initial voltage across the capacitor in Volts
    target_voltage : number or Engineer string
        The target voltage to reach in Volts
        
    Returns
    -------
    float
        Time in seconds
    """
    return rc_charge_time(resistance, capacitance, initial_voltage, 0.0, target_voltage)

@returns_unit("s")
@normalize_numeric_args
def rl_current_rise_time(resistance, inductance, final_current, target_current):
    """
    Calculate the time required for current through an inductor to rise
    from 0 to target_current when approaching final_current.
    
    t = -L/R × ln((final_current - target_current) / final_current)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
    final_current : number or Engineer string
        The final (asymptotic) current in Amperes
    target_current : number or Engineer string
        The target current to reach in Amperes
        
    Returns
    -------
    float
        Time in seconds
    """
    final_current = normalize_numeric(final_current)
    target_current = normalize_numeric(target_current)
    
    if abs(target_current - final_current) < 1e-12:
        return float('inf')  # Never reaches final current
    if final_current == 0:
        return 0.0  # No current flow
    
    return -(inductance / resistance) * np.log((final_current - target_current) / final_current)

@returns_unit("s")
@normalize_numeric_args
def rl_current_fall_time(resistance, inductance, initial_current, target_current):
    """
    Calculate the time required for current through an inductor to fall
    from initial_current to target_current (assuming decay to 0A).
    
    t = -L/R × ln(target_current / initial_current)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
    initial_current : number or Engineer string
        The initial current through the inductor in Amperes
    target_current : number or Engineer string
        The target current to reach in Amperes
        
    Returns
    -------
    float
        Time in seconds
    """
    initial_current = normalize_numeric(initial_current)
    target_current = normalize_numeric(target_current)
    
    if initial_current == 0:
        return 0.0  # No initial current
    if target_current == 0:
        return float('inf')  # Never reaches zero current
    
    return -(inductance / resistance) * np.log(target_current / initial_current)

@returns_unit("V")
@normalize_numeric_args
def rc_step_response(resistance, capacitance, time, initial_voltage=0, final_voltage=1):
    """
    Calculate the voltage across a capacitor at a given time after a step input.
    
    V(t) = final_voltage + (initial_voltage - final_voltage) × exp(-t / (R × C))
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    capacitance : number or Engineer string
        The capacitance value in Farads
    time : number or Engineer string
        The time at which to evaluate the response in seconds
    initial_voltage : number or Engineer string, optional
        The initial voltage across the capacitor in Volts (default: 0)
    final_voltage : number or Engineer string, optional
        The final voltage step in Volts (default: 1)
        
    Returns
    -------
    float
        Voltage across capacitor at time t in Volts
    """
    tau = resistance * capacitance
    return final_voltage + (initial_voltage - final_voltage) * np.exp(-time / tau)

@returns_unit("A")
@normalize_numeric_args
def rl_step_response(resistance, inductance, time, final_current=1):
    """
    Calculate the current through an inductor at a given time after a step input.
    
    I(t) = final_current × (1 - exp(-t × R / L))
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
    time : number or Engineer string
        The time at which to evaluate the response in seconds
    final_current : number or Engineer string, optional
        The final current step in Amperes (default: 1)
        
    Returns
    -------
    float
        Current through inductor at time t in Amperes
    """
    tau = inductance / resistance
    return final_current * (1 - np.exp(-time / tau))

@returns_unit("Hz")
@normalize_numeric_args
def rlc_resonant_frequency(inductance, capacitance):
    """
    Calculate the resonant frequency of an RLC circuit.
    
    f0 = 1 / (2π × sqrt(L × C))
    
    Parameters
    ----------
    inductance : number or Engineer string
        The inductance value in Henries
    capacitance : number or Engineer string
        The capacitance value in Farads
        
    Returns
    -------
    float
        Resonant frequency in Hz
    """
    return 1.0 / (2 * np.pi * np.sqrt(inductance * capacitance))

@normalize_numeric_args
def rlc_quality_factor(resistance, inductance, capacitance):
    """
    Calculate the quality factor (Q) of an RLC circuit.
    
    Q = (1/R) × sqrt(L/C)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
    capacitance : number or Engineer string
        The capacitance value in Farads
        
    Returns
    -------
    float
        Quality factor (dimensionless)
    """
    return (1.0 / resistance) * np.sqrt(inductance / capacitance)

@normalize_numeric_args
def rlc_damping_ratio(resistance, inductance, capacitance):
    """
    Calculate the damping ratio (ζ) of an RLC circuit.
    
    ζ = R/2 × sqrt(C/L)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
    capacitance : number or Engineer string
        The capacitance value in Farads
        
    Returns
    -------
    float
        Damping ratio (dimensionless)
        ζ < 1: underdamped
        ζ = 1: critically damped  
        ζ > 1: overdamped
    """
    return (resistance / 2.0) * np.sqrt(capacitance / inductance)

@returns_unit("Hz")
@normalize_numeric_args
def rlc_bandwidth(resistance, inductance):
    """
    Calculate the 3dB bandwidth of an RLC circuit.
    
    BW = R / (2π × L)
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    inductance : number or Engineer string
        The inductance value in Henries
        
    Returns
    -------
    float
        Bandwidth in Hz
    """
    return resistance / (2 * np.pi * inductance)