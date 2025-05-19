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
    "buck_regulator_inductor_peak_current", "buck_regulator_inductor_rms_current",
    "buck_regulator_min_capacitance_method1", "buck_regulator_min_capacitance_method2",
    "buck_regulator_min_capacitance_method3", "buck_regulator_min_capacitance",
    "buck_regulator_output_capacitor_max_esr", "buck_regulator_output_capacitor_rms_current",
    "buck_regulator_catch_diode_power",
]

def buck_regulator_inductance(vin, vout, frequency, ioutmax, K=0.3) -> Unit("H"):
    """
    Compute the optimal inductance for use in a buck regulator
    
    This formula is based on the the inductor ripple current fraction [K].
    
    The formula we use is:
    
    L = ((vin - vout) / (f * K * Ioutmax)) * (Vout/Vin)
    
    (note that Vout/Vin is an estimation for the duty cycle.)
    
    A good assumption which is shared by most major manufacturers is
    to choose the inductor value in between K=0.2 and K=0.4.
    Typically, the best inductor value is around K=0.3,
    but this depends on choice of inductor and the application.
    
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
    return ((vin - vout) / (frequency * K * ioutmax)) * (vout/vin)

InductorCurrent = namedtuple("InductorCurrent", ["peak", "rms", "ripple"])

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
    Compute an estimation for the peak, RMS & ripple inductor current.
    This does not include any safety factors
    
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
    return InductorCurrent(peak=Ilpeak, rms=Ilrms, ripple=ΔIL)

def buck_regulator_inductor_peak_current(vin, vout, inductance, frequency, ioutmax, safety_factor=1.0) -> Unit("A"):
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
    including the safety factor (default: 1.0).
    """
    return buck_regulator_inductor_current(
        vin, vout, inductance, frequency, ioutmax
    ).peak * safety_factor

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
    return buck_regulator_inductor_current(
        vin, vout, inductance, frequency, ioutmax
    ).rms * safety_factor
    

def buck_regulator_min_capacitance_method1(ripple_current, permissible_ripple_voltage, frequency):
    """
    Basic output capacitance calculation, based on the formula:
    C > 2*ΔIL / (fsw * ΔVout)
    where ΔIL is the inductor ripple current, fsw is the switching frequency,
    and ΔVout is the permissible ripple voltage.
    
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 35
    """
    ripple_current = normalize_numeric(ripple_current)
    permissible_ripple_voltage = normalize_numeric(permissible_ripple_voltage)
    frequency = normalize_numeric(frequency)
    return (2 * ripple_current) / (frequency * permissible_ripple_voltage)

def buck_regulator_min_capacitance_method2(inductance, nominal_output_voltage, output_voltage_ripple, max_load_current, light_load_current):
    """
    Compute the minimum capacitance required for a buck regulator
    based on the load current and the peak permissible output voltage.
    
    Cout > L * (Ioutmax² - Ioutmin²) / (Vpeak² - Vnom²)
    
    with Vpeak = Vnominal + output_voltage_ripple/2
    
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 36
    """
    inductance = normalize_numeric(inductance)
    nominal_output_voltage = normalize_numeric(nominal_output_voltage)
    output_voltage_ripple = normalize_numeric(output_voltage_ripple)
    max_load_current = normalize_numeric(max_load_current)
    # Compute secondary parameters
    peak_output_voltage = nominal_output_voltage + output_voltage_ripple / 2
    # Compute the minimum capacitance
    return inductance * (max_load_current**2 - light_load_current**2) / (peak_output_voltage**2 - nominal_output_voltage**2)

def buck_regulator_min_capacitance_method3(switching_frequency, output_voltage_ripple, ripple_current):
    """
    Compute the minimum capacitance required for a buck regulator
    based on the load current and the peak permissible output voltage.
    
    Cout > 1/(8 * fsw) * 1/ (ΔVout / ΔIL)
    
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 37
    """
    switching_frequency = normalize_numeric(switching_frequency)
    output_voltage_ripple = normalize_numeric(output_voltage_ripple)
    ripple_current = normalize_numeric(ripple_current)
    return 1 / (8 * switching_frequency) * 1 / (output_voltage_ripple / ripple_current)

def buck_regulator_min_capacitance(
    ripple_current,
    output_voltage_ripple,
    switching_frequency,
    inductance,
    nominal_output_voltage,
    max_load_current,
    light_load_current
) -> Unit("F"):
    """
    Calculate the minimum capacitance required for a buck regulator by taking
    the maximum of three different calculation methods.
    
    This conservative approach ensures all design constraints are met by using
    the largest capacitance value calculated from the three methods.
    
    Parameters
    ----------
    ripple_current : float
        The inductor ripple current (ΔIL)
    output_voltage_ripple : float
        The permissible output voltage ripple (ΔVout)
    switching_frequency : float
        The switching frequency of the regulator
    inductance : float
        The inductance value used for method 2
    nominal_output_voltage : float
        The nominal output voltage used for method 2
    max_load_current : float
        The maximum load current used for method 2
    light_load_current : float
        The light load current used for method 2
        
    Returns
    -------
    float
        The minimum required output capacitance in Farads
    """
    # Calculate using method 1
    c1 = buck_regulator_min_capacitance_method1(
        ripple_current, 
        output_voltage_ripple,
        switching_frequency
    )
    
    # Calculate using method 2
    c2 = buck_regulator_min_capacitance_method2(
        inductance, 
        nominal_output_voltage, 
        output_voltage_ripple, 
        max_load_current, 
        light_load_current
    )
    
    # Calculate using method 3
    c3 = buck_regulator_min_capacitance_method3(
        switching_frequency, 
        output_voltage_ripple, 
        ripple_current
    )
    
    # Return the maximum of all calculations
    return max(c1, c2, c3)

def buck_regulator_output_capacitor_max_esr(output_voltage_ripple, ripple_current):
    """
    Compute the maximum ESR of the output capacitor
    
    This is based on the formula:
    
    ESR < ΔVout / ΔIL
    
    where ΔVout is the permissible output voltage ripple,
    and ΔIL is the inductor ripple current.
    
    Returns the maximum ESR in Ohm
    
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 38
    """
    output_voltage_ripple = normalize_numeric(output_voltage_ripple)
    ripple_current = normalize_numeric(ripple_current)
    return output_voltage_ripple / ripple_current

def buck_regulator_output_capacitor_rms_current(
    input_voltage_max,
    output_voltage,
    inductance,
    switching_frequency,
):
    """
    Compute the RMS current rating of the output capacitor
    This is based on the formula:
    
    Irms = (Vout * (Vinmax-Vout)) / (sqrt(12) * Vinmax * L * fsw)
    
    where Vout is the output voltage, Vinmax is the maximum input voltage,
    L is the inductance, and fsw is the switching frequency.
        
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 39
    """
    input_voltage_max = normalize_numeric(input_voltage_max)
    output_voltage = normalize_numeric(output_voltage)
    inductance = normalize_numeric(inductance)
    switching_frequency = normalize_numeric(switching_frequency)
    
    return (output_voltage * (input_voltage_max - output_voltage)) / (
        (12**0.5) * input_voltage_max * inductance * switching_frequency
    )

def buck_regulator_catch_diode_power(vinmax, vout, iout, fsw, v_d="0.7V", c_j="200pF"):
    """
    Compute the minimum required power rating of the catch diode
    for non-synchronous buck regulators.
    
    P_D = ((Vinmax - Vout) * Iout * Vd) / (Vinmax) + (Cj * fsw * (Vin + Vd)²)/2
    where:
    * Vinmax is the maximum input voltage
    * Vout is the output voltage
    * Iout is the output current
    * Vd is the forward voltage drop of the diode
    * Cj is the junction capacitance of the diode (at Vinmax)
    * fsw is the switching frequency
    
    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 40
    """
    vinmax = normalize_numeric(vinmax)
    vout = normalize_numeric(vout)
    iout = normalize_numeric(iout)
    fsw = normalize_numeric(fsw)
    v_d = normalize_numeric(v_d)
    c_j = normalize_numeric(c_j)
    
    return ((vinmax - vout) * iout * v_d) / (vinmax) + (c_j * fsw * (vinmax + v_d)**2) / 2