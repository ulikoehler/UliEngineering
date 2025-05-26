#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for calculating time constants in electronic circuits
"""
import numpy as np
from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.Units import Unit

__all__ = [
    "rc_time_constant", "rl_time_constant", "rc_cutoff_frequency",
    "rl_cutoff_frequency", "rc_charge_time", "rc_discharge_time",
    "rl_current_rise_time", "rl_current_fall_time", "rc_step_response",
    "rl_step_response", "rlc_resonant_frequency", "rlc_quality_factor",
    "rlc_damping_ratio", "rlc_bandwidth"
]

def rc_time_constant(resistance, capacitance) -> Unit("s"):
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
    resistance = normalize_numeric(resistance)
    capacitance = normalize_numeric(capacitance)
    return resistance * capacitance

def rl_time_constant(resistance, inductance) -> Unit("s"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    return inductance / resistance

def rc_cutoff_frequency(resistance, capacitance) -> Unit("Hz"):
    """
    Calculate the cutoff frequency (fc) of an RC circuit.
    
    fc = 1 / (2π × R × C)
    
    This is the -3dB frequency where the output is 70.7% of the input
    in a low-pass or high-pass RC filter.
    
    Parameters
    ----------
    resistance : number or Engineer string
        The resistance value in Ohms
    capacitance : number or Engineer string
        The capacitance value in Farads
        
    Returns
    -------
    float
        Cutoff frequency in Hz
    """
    resistance = normalize_numeric(resistance)
    capacitance = normalize_numeric(capacitance)
    return 1.0 / (2 * np.pi * resistance * capacitance)

def rl_cutoff_frequency(resistance, inductance) -> Unit("Hz"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    return resistance / (2 * np.pi * inductance)

def rc_charge_time(resistance, capacitance, initial_voltage, final_voltage, target_voltage) -> Unit("s"):
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
    resistance = normalize_numeric(resistance)
    capacitance = normalize_numeric(capacitance)
    initial_voltage = normalize_numeric(initial_voltage)
    final_voltage = normalize_numeric(final_voltage)
    target_voltage = normalize_numeric(target_voltage)
    
    if abs(initial_voltage - final_voltage) < 1e-12:
        return 0.0  # Already at final voltage
    if abs(target_voltage - final_voltage) < 1e-12:
        return float('inf')  # Never reaches final voltage
    
    return -resistance * capacitance * np.log((target_voltage - final_voltage) / (initial_voltage - final_voltage))

def rc_discharge_time(resistance, capacitance, initial_voltage, target_voltage) -> Unit("s"):
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

def rl_current_rise_time(resistance, inductance, final_current, target_current) -> Unit("s"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    final_current = normalize_numeric(final_current)
    target_current = normalize_numeric(target_current)
    
    if abs(target_current - final_current) < 1e-12:
        return float('inf')  # Never reaches final current
    if final_current == 0:
        return 0.0  # No current flow
    
    return -(inductance / resistance) * np.log((final_current - target_current) / final_current)

def rl_current_fall_time(resistance, inductance, initial_current, target_current) -> Unit("s"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    initial_current = normalize_numeric(initial_current)
    target_current = normalize_numeric(target_current)
    
    if initial_current == 0:
        return 0.0  # No initial current
    if target_current == 0:
        return float('inf')  # Never reaches zero current
    
    return -(inductance / resistance) * np.log(target_current / initial_current)

def rc_step_response(resistance, capacitance, time, initial_voltage=0, final_voltage=1) -> Unit("V"):
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
    resistance = normalize_numeric(resistance)
    capacitance = normalize_numeric(capacitance)
    time = normalize_numeric(time)
    initial_voltage = normalize_numeric(initial_voltage)
    final_voltage = normalize_numeric(final_voltage)
    
    tau = resistance * capacitance
    return final_voltage + (initial_voltage - final_voltage) * np.exp(-time / tau)

def rl_step_response(resistance, inductance, time, final_current=1) -> Unit("A"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    time = normalize_numeric(time)
    final_current = normalize_numeric(final_current)
    
    tau = inductance / resistance
    return final_current * (1 - np.exp(-time / tau))

def rlc_resonant_frequency(inductance, capacitance) -> Unit("Hz"):
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
    inductance = normalize_numeric(inductance)
    capacitance = normalize_numeric(capacitance)
    return 1.0 / (2 * np.pi * np.sqrt(inductance * capacitance))

def rlc_quality_factor(resistance, inductance, capacitance) -> float:
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    capacitance = normalize_numeric(capacitance)
    return (1.0 / resistance) * np.sqrt(inductance / capacitance)

def rlc_damping_ratio(resistance, inductance, capacitance) -> float:
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    capacitance = normalize_numeric(capacitance)
    return (resistance / 2.0) * np.sqrt(capacitance / inductance)

def rlc_bandwidth(resistance, inductance) -> Unit("Hz"):
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
    resistance = normalize_numeric(resistance)
    inductance = normalize_numeric(inductance)
    return resistance / (2 * np.pi * inductance)