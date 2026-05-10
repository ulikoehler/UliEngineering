#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for computing switching regulator parameters."""
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO import normalize_numeric
from collections import namedtuple
from .Diode import normalize_voltage, VoltageV, normalize_current, CurrentA, normalize_resistance, ResistanceOhm
from .Filter import normalize_frequency, FrequencyHz, normalize_inductance, InductanceH
from .Capacitors import normalize_capacitance, CapacitanceFarad

__all__ = [
    "buck_regulator_inductance", "buck_regulator_inductor_current", "InductorCurrent",
    "buck_regulator_duty_cycle", "buck_regulator_inductor_ripple_current",
    "buck_regulator_inductor_peak_current", "buck_regulator_inductor_rms_current",
    "buck_regulator_min_capacitance_method1", "buck_regulator_min_capacitance_method2",
    "buck_regulator_min_capacitance_method3", "buck_regulator_min_capacitance",
    "buck_regulator_output_capacitor_max_esr", "buck_regulator_output_capacitor_rms_current",
    "buck_regulator_catch_diode_power", "buck_regulator_min_output_voltage",
    "buck_regulator_output_voltage_ripple",
]

@returns_unit("H")
def buck_regulator_inductance(vin: VoltageV, vout: VoltageV, frequency: FrequencyHz, ioutmax: CurrentA, K=0.3):
    """
    Compute the optimal inductance for use in a buck regulator.

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
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    ioutmax = normalize_current(ioutmax) if isinstance(ioutmax, str) else ioutmax
    return ((vin - vout) / (frequency * K * ioutmax)) * (vout/vin)

InductorCurrent = namedtuple("InductorCurrent", ["peak", "rms", "ripple"])
RippleVoltage = namedtuple("RippleVoltage", ["pp", "rms", "capacitive_pp", "esr_pp"])

def buck_regulator_duty_cycle(vin: VoltageV, vout: VoltageV) -> float:
    """Estimate the duty cycle of a buck regulator.

    D = Vout/Vin

    """
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    return vout / vin

@returns_unit("A")
def buck_regulator_inductor_ripple_current(vin: VoltageV, vout: VoltageV, inductance: InductanceH, frequency: FrequencyHz, ioutmax: CurrentA):
    """
    Compute the ripple current ΔIL in the inductor.

    This can be used to determine the peak current rating of the inductor.

    The formula is:

    ΔIL = (Vin - Vout) * D / (L * frequency)
    where D = Vout/Vin

    Returns the ripple current in Amperes

    """
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    ioutmax = normalize_current(ioutmax) if isinstance(ioutmax, str) else ioutmax
    D = buck_regulator_duty_cycle(vin, vout)
    return (vin - vout) * D / (inductance * frequency)

def buck_regulator_inductor_current(vin: VoltageV, vout: VoltageV, inductance: InductanceH, frequency: FrequencyHz, ioutmax: CurrentA) -> InductorCurrent:
    """
    Compute an estimation for the peak, RMS & ripple inductor current.

    This does not include any safety factors.

    This can be used to determine inductor value.

    This approach is based on the formula found in the LM76002 datasheet
    from Texas instruments:
    https://www.ti.com/lit/ds/symlink/lm76002.pdf

    D = (Vout/Vin) # Duty cycle estimation
    ΔIL = (Vin - Vout) * D / (L * frequency)

    Ilpeak = Ioutmax + ΔIL / 2
    Ilrms = sqrt(Ioutmax^2 + ΔIL^2 / 12)

    Returns an InductorCurrent namedtuple with the peak and RMS current (unit: Amperes)

    """
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    ioutmax = normalize_current(ioutmax) if isinstance(ioutmax, str) else ioutmax
    ΔIL = buck_regulator_inductor_ripple_current(vin, vout, inductance, frequency, ioutmax)
    Ilpeak = ioutmax + ΔIL / 2
    Ilrms = (ioutmax**2 + ΔIL**2 / 12)**0.5
    return InductorCurrent(peak=Ilpeak, rms=Ilrms, ripple=ΔIL)

@returns_unit("A")
def buck_regulator_inductor_peak_current(vin: VoltageV, vout: VoltageV, inductance: InductanceH, frequency: FrequencyHz, ioutmax: CurrentA, safety_factor=1.0):
    """Compute the peak inductor current rating.

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
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    ioutmax = normalize_current(ioutmax) if isinstance(ioutmax, str) else ioutmax
    return buck_regulator_inductor_current(
        vin, vout, inductance, frequency, ioutmax
    ).peak * safety_factor

@returns_unit("A")
def buck_regulator_inductor_rms_current(vin: VoltageV, vout: VoltageV, inductance: InductanceH, frequency: FrequencyHz, ioutmax: CurrentA, safety_factor=1.2):
    """
    Compute the RMS inductor current rating.

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
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    ioutmax = normalize_current(ioutmax) if isinstance(ioutmax, str) else ioutmax
    return buck_regulator_inductor_current(
        vin, vout, inductance, frequency, ioutmax
    ).rms * safety_factor


@returns_unit("F")
def buck_regulator_min_capacitance_method1(ripple_current: CurrentA, permissible_ripple_voltage: VoltageV, frequency: FrequencyHz):
    """
    Compute the basic output capacitance.

    Based on the formula: C > 2*ΔIL / (fsw * ΔVout) where ΔIL is the inductor
    ripple current, fsw is the switching frequency, and ΔVout is the permissible
    ripple voltage.

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf Formula 35.

    """
    ripple_current = normalize_current(ripple_current) if isinstance(ripple_current, str) else ripple_current
    permissible_ripple_voltage = normalize_voltage(permissible_ripple_voltage) if isinstance(permissible_ripple_voltage, str) else permissible_ripple_voltage
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    return (2 * ripple_current) / (frequency * permissible_ripple_voltage)

@returns_unit("F")
def buck_regulator_min_capacitance_method2(inductance: InductanceH, nominal_output_voltage: VoltageV, output_voltage_ripple: VoltageV, max_load_current: CurrentA, light_load_current: CurrentA):
    """
    Compute the minimum capacitance required for a buck regulator.

    Based on the load current and the peak permissible output voltage.

    Cout > L * (Ioutmax² - Ioutmin²) / (Vpeak² - Vnom²) with Vpeak = Vnominal + output_voltage_ripple/2

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf Formula 36.

    """
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    nominal_output_voltage = normalize_voltage(nominal_output_voltage) if isinstance(nominal_output_voltage, str) else nominal_output_voltage
    output_voltage_ripple = normalize_voltage(output_voltage_ripple) if isinstance(output_voltage_ripple, str) else output_voltage_ripple
    max_load_current = normalize_current(max_load_current) if isinstance(max_load_current, str) else max_load_current
    light_load_current = normalize_current(light_load_current) if isinstance(light_load_current, str) else light_load_current
    # Compute secondary parameters
    peak_output_voltage = nominal_output_voltage + output_voltage_ripple / 2
    # Compute the minimum capacitance
    return inductance * (max_load_current**2 - light_load_current**2) / (peak_output_voltage**2 - nominal_output_voltage**2)

@returns_unit("F")
def buck_regulator_min_capacitance_method3(switching_frequency: FrequencyHz, output_voltage_ripple: VoltageV, ripple_current: CurrentA):
    """
    Compute the minimum capacitance required for a buck regulator.

    Based on the load current and the peak permissible output voltage.

    Cout > 1/(8 * fsw) * 1/ (ΔVout / ΔIL)

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf Formula 37.

    """
    switching_frequency = normalize_frequency(switching_frequency) if isinstance(switching_frequency, str) else switching_frequency
    output_voltage_ripple = normalize_voltage(output_voltage_ripple) if isinstance(output_voltage_ripple, str) else output_voltage_ripple
    ripple_current = normalize_current(ripple_current) if isinstance(ripple_current, str) else ripple_current
    return 1 / (8 * switching_frequency) * 1 / (output_voltage_ripple / ripple_current)

@returns_unit("F")
def buck_regulator_min_capacitance(
    ripple_current: CurrentA,
    output_voltage_ripple: VoltageV,
    switching_frequency: FrequencyHz,
    inductance: InductanceH,
    nominal_output_voltage: VoltageV,
    max_load_current: CurrentA,
    light_load_current: CurrentA
):
    """
    Calculate the minimum capacitance required for a buck regulator.

    Take the maximum of three different calculation methods. This conservative
    approach ensures all design constraints are met by using the largest capacitance
    value calculated from the three methods.

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

@returns_unit("Ω")
def buck_regulator_output_capacitor_max_esr(output_voltage_ripple: VoltageV, ripple_current: CurrentA):
    """Compute the maximum ESR of the output capacitor.

    This is based on the formula:

    ESR < ΔVout / ΔIL

    where ΔVout is the permissible output voltage ripple,
    and ΔIL is the inductor ripple current.

    Returns the maximum ESR in Ohm

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf
    Formula 38

    """
    output_voltage_ripple = normalize_voltage(output_voltage_ripple) if isinstance(output_voltage_ripple, str) else output_voltage_ripple
    ripple_current = normalize_current(ripple_current) if isinstance(ripple_current, str) else ripple_current
    return output_voltage_ripple / ripple_current

@returns_unit("A")
def buck_regulator_output_capacitor_rms_current(
    input_voltage_max: VoltageV,
    output_voltage: VoltageV,
    inductance: InductanceH,
    switching_frequency: FrequencyHz,
):
    """
    Compute the RMS current rating of the output capacitor.

    This is based on the formula:

    Irms = (Vout * (Vinmax-Vout)) / (sqrt(12) * Vinmax * L * fsw)

    where Vout is the output voltage, Vinmax is the maximum input voltage,
    L is the inductance, and fsw is the switching frequency.

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf Formula 39.

    """
    input_voltage_max = normalize_voltage(input_voltage_max) if isinstance(input_voltage_max, str) else input_voltage_max
    output_voltage = normalize_voltage(output_voltage) if isinstance(output_voltage, str) else output_voltage
    inductance = normalize_inductance(inductance) if isinstance(inductance, str) else inductance
    switching_frequency = normalize_frequency(switching_frequency) if isinstance(switching_frequency, str) else switching_frequency
    return (output_voltage * (input_voltage_max - output_voltage)) / (
        (12**0.5) * input_voltage_max * inductance * switching_frequency
    )

@returns_unit("W")
def buck_regulator_catch_diode_power(vinmax: VoltageV, vout: VoltageV, iout: CurrentA, fsw: FrequencyHz, v_d="0.7V", c_j="200pF"):
    """
    Compute the minimum required power rating of the catch diode.

    For non-synchronous buck regulators.

    P_D = ((Vinmax - Vout) * Iout * Vd) / (Vinmax) + (Cj * fsw * (Vin + Vd)²)/2
    where:
    * Vinmax is the maximum input voltage
    * Vout is the output voltage
    * Iout is the output current
    * Vd is the forward voltage drop of the diode
    * Cj is the junction capacitance of the diode (at Vinmax)
    * fsw is the switching frequency

    Source: https://www.ti.com/lit/ds/symlink/tps54561.pdf Formula 40.

    """
    vinmax = normalize_voltage(vinmax) if isinstance(vinmax, str) else vinmax
    vout = normalize_voltage(vout) if isinstance(vout, str) else vout
    iout = normalize_current(iout) if isinstance(iout, str) else iout
    fsw = normalize_frequency(fsw) if isinstance(fsw, str) else fsw
    v_d = normalize_voltage(v_d) if isinstance(v_d, str) else v_d
    c_j = normalize_capacitance(c_j) if isinstance(c_j, str) else c_j
    return ((vinmax - vout) * iout * v_d) / (vinmax) + (c_j * fsw * (vinmax + v_d)**2) / 2

@returns_unit("V")
def buck_regulator_min_output_voltage(vin: VoltageV, t_on_min, switching_frequency: FrequencyHz):
    """Compute the minimum output voltage of a buck regulator given its minimum on time.

    The formula is:
        Vout_min = Vin * (t_on_min * f_sw)
    where:
        Vin = input voltage
        t_on_min = minimum on time (seconds)
        f_sw = switching frequency (Hz)

    Returns the minimum output voltage in the same units as Vin.

    """
    vin = normalize_voltage(vin) if isinstance(vin, str) else vin
    t_on_min = normalize_numeric(t_on_min) if isinstance(t_on_min, str) else t_on_min
    switching_frequency = normalize_frequency(switching_frequency) if isinstance(switching_frequency, str) else switching_frequency
    return vin * t_on_min * switching_frequency

def buck_regulator_output_voltage_ripple(ripple_current: CurrentA, frequency: FrequencyHz, capacitance: CapacitanceFarad, esr: ResistanceOhm = 0.0) -> RippleVoltage:
    """Compute the output voltage ripple breakdown for a buck regulator.

    This function calculates the peak-to-peak and RMS ripple, providing
    the individual contributions from both the capacitance and the ESR.

    ### Reasoning for the Formula:

    1. **Capacitive Peak-to-Peak (ΔVout_C):**
       The inductor current ripple (ΔIL) is a triangular waveform. The charge (ΔQ)
       delivered to the capacitor is the area of the triangle above the average
       current.
       ΔQ = (1/2) * (ΔIL / 2) * (T / 2) = ΔIL / (8 * frequency)
       Using ΔV = ΔQ / C:
       ΔVout_C = ΔIL / (8 * frequency * capacitance)

    2. **ESR Peak-to-Peak (ΔVout_ESR):**
       Derived from Ohm's Law as the ripple current passes through the
       internal resistance:
       ΔVout_ESR = ΔIL * ESR

    3. **Total Peak-to-Peak (pp):**
       In the worst case (where peaks align), these are summed:
       ΔVout_total = ΔVout_C + ΔVout_ESR

    4. **RMS Ripple (rms):**
       The ripple waveform in a buck converter is essentially triangular.
       For a triangular wave, the RMS value of the AC component is:
       V_rms = ΔVout_total / (2 * sqrt(3))

    Parameters
    ----------
    ripple_current : float
        The inductor ripple current (ΔIL) in Amperes.
    frequency : float
        The switching frequency (fsw) in Hertz.
    capacitance : float
        The output capacitance (Cout) in Farads.
    esr : float, optional
        The Equivalent Series Resistance in Ohms (default: 0.0).

    Returns
    -------
    RippleVoltage
        A namedtuple containing:
        - pp: Total peak-to-peak ripin=Vin, Vout=Vout, L=inductor, fsw=fsw, Iout=Ioutmax, Cout=Coutple voltage (V)
        - rms: Estimated RMS ripple voltage (V)
        - capacitive_pp: P-P ripple from capacitance only (V)
        - esr_pp: P-P ripple from ESR only (V)

    """
    ripple_current = normalize_current(ripple_current) if isinstance(ripple_current, str) else ripple_current
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    capacitance = normalize_capacitance(capacitance) if isinstance(capacitance, str) else capacitance
    esr = normalize_resistance(esr) if isinstance(esr, str) else esr
    # Calculate P-P components
    cap_pp = ripple_current / (8 * frequency * capacitance)
    esr_pp = ripple_current * esr
    total_pp = cap_pp + esr_pp

    # Calculate RMS (Triangular approximation)
    # RMS = V_pp / (2 * sqrt(3))
    rms = total_pp / (2 * (3**0.5))

    return RippleVoltage(
        pp=total_pp,
        rms=rms,
        capacitive_pp=cap_pp,
        esr_pp=esr_pp
    )