#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for LED calculations."""
from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.Exceptions import OperationImpossibleException
from UliEngineering.Electronics.Resistors import resistor_current_by_power
from .Diode import normalize_voltage, VoltageV, normalize_current, CurrentA, normalize_resistance, ResistanceOhm, normalize_power, PowerW

__all__ = [
    "LEDForwardVoltages",
    "led_series_resistor",
    "led_series_resistor_power",
    "led_series_resistor_maximum_current",
    "led_series_resistor_current",
]

class LEDForwardVoltages:

    """Common LED forward voltage values.
    Source: http://www.elektronik-kompendium.de/sites/bau/1109111.htm.
    NOTE: These do NOT necessarily represent the actual forward voltages
    of any LED you choose but rather the typical forward voltage at nominal
    current.

    Note that diode testers test the forward voltage with rather low currents
    and the forward voltage might vary slightly at operating current.
    Take that into account when operating a LED near its maximum allowed current.

    """
    Infrared = 1.5
    Red = 1.6
    Yellow = 2.2
    Green = 2.1
    Blue = 2.9
    White = 4.0

@returns_unit("Ω")
def led_series_resistor(vsupply: VoltageV, ioperating: CurrentA, vforward: VoltageV):
    """
    Compute the required series resistor for operating a LED with forward
    voltage vforward at current ioperating on a supply voltage of vsupply.

    Tolerances are not taken into account.

    Parameters
    ----------
    vsupply : VoltageV
        Supply voltage in Volts.
    ioperating : CurrentA
        Operating current in Amperes.
    vforward : VoltageV
        Forward voltage of the LED in Volts.

    Returns
    -------
    float
        Required series resistor value in Ohms.
    """
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    ioperating = normalize_current(ioperating) if isinstance(ioperating, str) else ioperating
    vforward = normalize_voltage(vforward) if isinstance(vforward, str) else vforward
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    return (vsupply - vforward) / ioperating

@returns_unit("W")
def led_series_resistor_power(vsupply: VoltageV, ioperating: CurrentA, vforward: VoltageV):
    """
    Compute the required series resistor power for operating a LED with
    forward voltage vforward at current ioperating on a supply voltage of
    vsupply.

    The resulting power value is the minimum rated value for the resistor
    for continuous operation.

    Tolerances are not taken into account.

    Parameters
    ----------
    vsupply : VoltageV
        Supply voltage in Volts.
    ioperating : CurrentA
        Operating current in Amperes.
    vforward : VoltageV
        Forward voltage of the LED in Volts.

    Returns
    -------
    float
        Required resistor power in Watts.
    """
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    ioperating = normalize_current(ioperating) if isinstance(ioperating, str) else ioperating
    vforward = normalize_voltage(vforward) if isinstance(vforward, str) else vforward
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    # Will raise OperationImpossibleException if vforward > vsupply
    resistor_value = led_series_resistor(vsupply, ioperating, vforward)
    return resistor_value * ioperating * ioperating

@returns_unit("A")
def led_series_resistor_maximum_current(resistance: ResistanceOhm, power_rating: PowerW):
    """
    Compute the maximum current through a LED + series resistor combination,
    so that the power rating of the resistor is not exceeded (i.e. the current
    where the dissipated power is exactly the power rating).

    Tolerances are not taken into account.

    Parameters
    ----------
    resistance : ResistanceOhm
        Series resistor value in Ohms.
    power_rating : PowerW
        Power rating of the resistor in Watts.

    Returns
    -------
    float
        Maximum current in Amperes.
    """
    power_rating = normalize_power(power_rating) if isinstance(power_rating, str) else power_rating
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    # Compute the current that would flow through the resistor
    current = resistor_current_by_power(resistance, power_rating)
    return current

@returns_unit("A")
def led_series_resistor_current(vsupply: VoltageV, resistance: ResistanceOhm, vforward: VoltageV):
    """
    Compute the current that flows through a LED + series resistor combination
    when connected to a supply voltage vsupply and a series resistor of
    resistance.

    Tolerances are not taken into account.

    Parameters
    ----------
    vsupply : VoltageV
        Supply voltage in Volts.
    resistance : ResistanceOhm
        Series resistor value in Ohms.
    vforward : VoltageV
        Forward voltage of the LED in Volts.

    Returns
    -------
    float
        Current through the LED in Amperes.
    """
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    resistance = normalize_resistance(resistance) if isinstance(resistance, str) else resistance
    vforward = normalize_voltage(vforward) if isinstance(vforward, str) else vforward
    if vforward > vsupply:
        raise OperationImpossibleException(
            f"Can't operate LED with forward voltage {vforward} on {vsupply} supply"
        )
    return (vsupply - vforward) / resistance
