#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities to calculate MOSFET parameters."""
from typing import Annotated

from UliEngineering.EngineerIO.Decorators import returns_unit
from UliEngineering.EngineerIO.Types import NormalizableArgument, NormalizedComputable
from UliEngineering.Physics._normalize import normalize_with_known_units
from .Diode import normalize_voltage, VoltageV
from .Filter import normalize_frequency, FrequencyHz

__all__ = [
    "mosfet_gate_charge_losses", "mosfet_gate_charge_loss_per_cycle",
    "mosfet_gate_capacitance_from_gate_charge",
    "normalize_charge", "ChargeC",
]


def normalize_charge(Q: NormalizableArgument) -> NormalizedComputable:
    """Normalize charge to coulombs."""
    return normalize_with_known_units(Q, {"C": 1.0, "mC": 1e-3, "µC": 1e-6, "nC": 1e-9}, quantity_name="charge")

ChargeC = Annotated[NormalizedComputable, normalize_charge]

@returns_unit("W")
def mosfet_gate_charge_losses(total_gate_charge: ChargeC, vsupply: VoltageV, frequency: FrequencyHz = "100 kHz"):
    """Compute the gate charge loss of a MOSFET in a switch-mode power-supply application as a total power (integrated per second).

    Ref:
    http://rohmfs.rohm.com/en/products/databook/applinote/ic/power/switching_regulator/power_loss_appli-e.pdf

    Parameters
    ----------
    total_gate_charge : ChargeC
        The total gate charge in Coulomb.
        For multiple MOSFETs such as in synchronous applications,
        add their gate charges together.
    vsupply : VoltageV
        The gate driver supply voltage in Volts.
    frequency : FrequencyHz
        The switching frequency in Hz.

    Returns
    -------
    float
        Gate charge loss in Watts.
    
    """
    total_gate_charge = normalize_charge(total_gate_charge) if isinstance(total_gate_charge, str) else total_gate_charge
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    frequency = normalize_frequency(frequency) if isinstance(frequency, str) else frequency
    return mosfet_gate_charge_loss_per_cycle(total_gate_charge, vsupply) * frequency


@returns_unit("J")
def mosfet_gate_charge_loss_per_cycle(total_gate_charge: ChargeC, vsupply: VoltageV):
    """Compute the gate charge loss of a MOSFET in a switch-mode power-supply
    
    application per switching cycle.

    Ref:
    http://rohmfs.rohm.com/en/products/databook/applinote/ic/power/switching_regulator/power_loss_appli-e.pdf

    Parameters
    ----------
    total_gate_charge : ChargeC
        The total gate charge in Coulomb.
        For multiple MOSFETs such as in synchronous applications,
        add their gate charges together.
    vsupply : VoltageV
        The gate driver supply voltage in Volts.

    Returns
    -------
    float
        Gate charge loss per cycle in Joules.
    
    """
    total_gate_charge = normalize_charge(total_gate_charge) if isinstance(total_gate_charge, str) else total_gate_charge
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    return total_gate_charge * vsupply

@returns_unit("F")
def mosfet_gate_capacitance_from_gate_charge(total_gate_charge: ChargeC, vsupply: VoltageV):
    """Compute the gate capacitance of a MOSFET in a switch-mode power-supply
    
    application.

    Parameters
    ----------
    total_gate_charge : ChargeC
        The total gate charge in Coulomb.
        For multiple MOSFETs such as in synchronous applications,
        add their gate charges together.
    vsupply : VoltageV
        The gate driver supply voltage in Volts.

    Returns
    -------
    float
        Gate capacitance in Farads.
    
    """
    total_gate_charge = normalize_charge(total_gate_charge) if isinstance(total_gate_charge, str) else total_gate_charge
    vsupply = normalize_voltage(vsupply) if isinstance(vsupply, str) else vsupply
    return total_gate_charge / vsupply