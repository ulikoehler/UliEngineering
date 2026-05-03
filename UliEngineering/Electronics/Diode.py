#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for diode calculations using the Shockley diode equation.
"""
from typing import cast

from scipy.constants import elementary_charge, k as boltzmann_k
from scipy.special import lambertw

import numpy as np

from UliEngineering.EngineerIO import normalize_numeric
from UliEngineering.EngineerIO.Decorators import normalize_numeric_args, returns_unit
from UliEngineering.Physics.Temperature import normalize_temperature_kelvin

__all__ = [
    "DiodeModel",
    "SimpleDiodeModel",
    "ShockleyDiodeModel",
    "normalize_diode_model",
    "diode_thermal_voltage",
    "shockley_diode_current",
    "shockley_diode_voltage",
    "shockley_diode_saturation_current",
    "shockley_diode_small_signal_resistance",
    "shockley_diode_power",
]


def _validate_positive(name, value):
    if np.any(value <= 0):
        raise ValueError(f"{name} must be greater than zero")


class DiodeModel:
    """
    Base class for diode models that support analytic RC timing calculations.
    """

    def minimum_series_voltage(self):
        raise NotImplementedError()

    def forward_voltage(self, current):
        raise NotImplementedError()

    def series_current(self, total_voltage, resistance):
        raise NotImplementedError()

    def series_current_integral(self, total_voltage, resistance):
        raise NotImplementedError()


class SimpleDiodeModel(DiodeModel):
    """
    Constant forward-voltage diode model.
    """

    def __init__(self, forward_voltage="0V"):
        self.forward_voltage_drop = normalize_numeric(forward_voltage)

    def minimum_series_voltage(self):
        return self.forward_voltage_drop

    def forward_voltage(self, current):
        current = normalize_numeric(current)
        return np.where(np.asarray(current) > 0, self.forward_voltage_drop, 0.0)

    def series_current(self, total_voltage, resistance):
        total_voltage = normalize_numeric(total_voltage)
        resistance = normalize_numeric(resistance)
        with np.errstate(divide="ignore", invalid="ignore"):
            current = np.divide(total_voltage - self.forward_voltage_drop, resistance)
        return np.where(np.asarray(total_voltage) > self.forward_voltage_drop, current, 0.0)

    def series_current_integral(self, total_voltage, resistance):
        total_voltage = normalize_numeric(total_voltage)
        resistance = normalize_numeric(resistance)
        with np.errstate(divide="ignore", invalid="ignore"):
            return resistance * np.log(total_voltage - self.forward_voltage_drop)


class ShockleyDiodeModel(DiodeModel):
    """
    Shockley diode model with analytic series resistor solutions.
    """

    def __init__(self, saturation_current, ideality_factor=1.0, temperature="25°C"):
        self.saturation_current = normalize_numeric(saturation_current)
        self.ideality_factor = normalize_numeric(ideality_factor)
        self.temperature = temperature
        _validate_positive("saturation_current", self.saturation_current)
        _validate_positive("ideality_factor", self.ideality_factor)

    @property
    def voltage_scale(self):
        return _shockley_voltage_scale(self.ideality_factor, self.temperature)

    def minimum_series_voltage(self):
        return 0.0

    def forward_voltage(self, current):
        return shockley_diode_voltage(
            current,
            self.saturation_current,
            ideality_factor=self.ideality_factor,
            temperature=self.temperature,
        )

    def _lambert_terms(self, total_voltage, resistance):
        total_voltage = normalize_numeric(total_voltage)
        resistance = normalize_numeric(resistance)
        k = resistance * self.saturation_current / self.voltage_scale
        with np.errstate(over="ignore", invalid="ignore"):
            arg = k * np.exp((total_voltage + resistance * self.saturation_current) / self.voltage_scale)
            z = np.real(lambertw(arg))
        return k, z

    def series_current(self, total_voltage, resistance):
        resistance = normalize_numeric(resistance)
        _, z = self._lambert_terms(total_voltage, resistance)
        return self.voltage_scale * z / resistance - self.saturation_current

    def series_current_integral(self, total_voltage, resistance):
        resistance = normalize_numeric(resistance)
        k, z = self._lambert_terms(total_voltage, resistance)
        with np.errstate(divide="ignore", invalid="ignore"):
            return resistance * (((k + 1.0) / k) * np.log(z - k) - (1.0 / k) * np.log(z))


def normalize_diode_model(model):
    if model is None:
        return SimpleDiodeModel(0.0)
    if isinstance(model, DiodeModel):
        return model
    return SimpleDiodeModel(model)


def _shockley_voltage_scale(ideality_factor, temperature):
    _validate_positive("ideality_factor", ideality_factor)
    return ideality_factor * diode_thermal_voltage(temperature)


@returns_unit("V")
def diode_thermal_voltage(temperature="25°C"):
    """
    Compute the thermal voltage $V_T = kT/q$ of a diode.

    Parameters:
    - temperature: The junction temperature.

    Returns:
    The thermal voltage in volts.
    """
    temperature_kelvin = cast(float, normalize_temperature_kelvin(temperature))
    return boltzmann_k * temperature_kelvin / elementary_charge


@returns_unit("A")
@normalize_numeric_args(exclude=["temperature"])
def shockley_diode_current(voltage, saturation_current, ideality_factor=1.0, temperature="25°C"):
    """
    Compute the diode current using the Shockley diode equation.

    Parameters:
    - voltage: The diode voltage in volts.
    - saturation_current: The diode saturation current in amperes.
    - ideality_factor: The ideality factor n.
    - temperature: The junction temperature.

    Returns:
    The diode current in amperes.
    """
    _validate_positive("saturation_current", saturation_current)
    voltage_scale = _shockley_voltage_scale(ideality_factor, temperature)
    return saturation_current * np.expm1(voltage / voltage_scale)


@returns_unit("V")
@normalize_numeric_args(exclude=["temperature"])
def shockley_diode_voltage(current, saturation_current, ideality_factor=1.0, temperature="25°C"):
    """
    Compute the diode voltage from the Shockley diode equation.

    Parameters:
    - current: The diode current in amperes.
    - saturation_current: The diode saturation current in amperes.
    - ideality_factor: The ideality factor n.
    - temperature: The junction temperature.

    Returns:
    The diode voltage in volts.
    """
    _validate_positive("saturation_current", saturation_current)
    if np.any(current <= -saturation_current):
        raise ValueError("current must be greater than -saturation_current")
    voltage_scale = _shockley_voltage_scale(ideality_factor, temperature)
    return voltage_scale * np.log1p(current / saturation_current)


@returns_unit("A")
@normalize_numeric_args(exclude=["temperature"])
def shockley_diode_saturation_current(voltage, current, ideality_factor=1.0, temperature="25°C"):
    """
    Compute the saturation current from one operating point.

    Parameters:
    - voltage: The diode voltage in volts.
    - current: The diode current in amperes.
    - ideality_factor: The ideality factor n.
    - temperature: The junction temperature.

    Returns:
    The diode saturation current in amperes.
    """
    if np.any(voltage == 0):
        raise ValueError("voltage must be non-zero to infer saturation current")
    voltage_scale = _shockley_voltage_scale(ideality_factor, temperature)
    return current / np.expm1(voltage / voltage_scale)


@returns_unit("Ω")
@normalize_numeric_args(exclude=["temperature"])
def shockley_diode_small_signal_resistance(current, ideality_factor=1.0, temperature="25°C"):
    """
    Compute the small-signal resistance $r_d = nV_T/I$ of a diode.

    Parameters:
    - current: The diode current in amperes.
    - ideality_factor: The ideality factor n.
    - temperature: The junction temperature.

    Returns:
    The small-signal resistance in ohms.
    """
    voltage_scale = _shockley_voltage_scale(ideality_factor, temperature)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(voltage_scale, current)


@returns_unit("W")
@normalize_numeric_args(exclude=["temperature"])
def shockley_diode_power(voltage, saturation_current, ideality_factor=1.0, temperature="25°C"):
    """
    Compute the power dissipated by a diode from the Shockley equation.

    Parameters:
    - voltage: The diode voltage in volts.
    - saturation_current: The diode saturation current in amperes.
    - ideality_factor: The ideality factor n.
    - temperature: The junction temperature.

    Returns:
    The diode power in watts.
    """
    return voltage * shockley_diode_current(voltage, saturation_current, ideality_factor, temperature)